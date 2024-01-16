import os
import json
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from torch.optim import AdamW
import math
from visdom import Visdom
from conformer import ConformerBlock


# Dataset
class myDataset(Dataset):
    def __init__(self, data_dir, segment_len=128):
        self.data_dir = data_dir
        self.segment_len = segment_len

        # Load the mapping from speaker name to their corresponding id.
        mapping_path = Path(data_dir) / "mapping.json"
        # mapping_path = Path(os.path.join(data_dir, "mapping.json"))
        mapping = json.load(mapping_path.open())
        self.speaker2id = mapping["speaker2id"]

        # Load metadata of training data.
        metadata_path = Path(data_dir) / "metadata.json"
        # metadata_path = Path(os.path.join(data_dir, "metadata.json"))
        metadata = json.load(open(metadata_path))["speakers"]  # Select the value of key "speakers".

        # Get the total number of speaker.
        self.speaker_num = len(metadata.keys())
        self.data = []
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances["feature_path"], self.speaker2id[speaker]])

    def __len__(self):
        return len(self.data)  # uttr文件的个数

    def __getitem__(self, index):
        # Corresponding to self.data.append([utterances["feature_path"], self.speaker2id[speaker]]).
        feat_path, speaker = self.data[index]
        # Load preprocessed mel-spectrogram which is saved as PyTorch pt file.
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        # Segment mel-spectrogram into "segment_len" frames.
        # 即实际从Dataset中返回的每个uttr的最大长度是segment_len
        if len(mel) > self.segment_len:
            # Randomly get the starting point of the segment.
            start = random.randint(0, len(mel) - self.segment_len)
            # Get a segment with "segment_len" frames.
            mel = torch.FloatTensor(mel[start:start + self.segment_len])
        else:
            mel = torch.FloatTensor(mel)
        # Turn the speaker id into long for computing loss later.
        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker

    def get_speaker_number(self):
        return self.speaker_num


# Dataloader
def collate_batch(batch):
    """
    Process features within a batch, collate a batch of data.
    由于在Dataset中将每个uttr文件的最大长度限制为segment_len，因此uttr数据长度不一，需要进行pad操作pad
    """
    mel, speaker = zip(*batch)
    # Because we train the model batch by batch,
    # we need to pad the features in the same batch to make their lengths the same.
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)  # pad with log 10^(-20) which is very small value.
    # mel: (batch size, length, 40)
    return mel, torch.FloatTensor(speaker).long()


def get_dataloader(data_dir, batch_size, n_workers):
    """Generate dataloader"""
    dataset = myDataset(data_dir)
    speaker_num = dataset.get_speaker_number()
    # Split dataset into training dataset and validation dataset
    trainlen = int(0.9 * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]
    trainset, validset = random_split(dataset, lengths)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    return train_loader, valid_loader, speaker_num


# Model
class Classifier(nn.Module):
    def __init__(self, d_model=256, n_spks=600, dropout=0.1):
        super().__init__()
        # 输入维度是梅尔频谱的特征维度40，将输入维度映射到d_model维度
        self.prenet = nn.Linear(40, d_model)
        # Conformer optimization
        self.conformer_layer = ConformerBlock(
            dim=d_model,
            dim_head=256,
            heads=1,  # set 1
            ff_mult=4,
            conv_expansion_factor=18,
            conv_kernel_size=41,
            attn_dropout=dropout,
            ff_dropout=dropout,
            conv_dropout=dropout
        )

        # 分类预测层：将特征从d_model维度映射到speaker_num的维度
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        """
        input args:
            mels: (batch size, length, 40)
        return:
            out: (batch size, n_spks)
        """
        # 输入的梅尔频谱形状是(batch size, length, 40)
        out = self.prenet(mels)  # out: (batch size, length, d_model)
        # encoder layer期望的输入形状是(length, batch size, d_model)，因此需要permute重整形状
        out = out.permute(1, 0, 2)  # out: (length, batch size, d_model)

        # 使用Conformer优化
        out = self.conformer_layer(out)  # out: (length, batch size, d_model)
        out = out.permute(1, 0, 2)  # out = out.transpose(0, 1)二者等效，调整out形状为(batch size, length, d_model)

        # 使用mean pooling保留频率维度，在时间维度上求平均以整合
        stats = out.mean(dim=1)  # status: (batch size, d_model)

        out = self.pred_layer(stats)  # out: (batch, n_spks)
        return out


# 学习率的预热与余弦退火
def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,  # 优化器
        num_warmup_steps: int,  # 预热阶段的总步数
        num_training_steps: int,  # 训练阶段的总步数
        num_cycles: float = 0.5,  # 余弦退火的周期，周期越大，学习率的余弦衰减越缓慢
        last_epoch: int = -1,
):  # 最终Optimizer真正用来更新参数的学习率的值是lr_lambda返回的结果与初始化Optimizer时设定的lr的乘积。
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        # 首先求出目前在退火过程中的进度比例
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# Model Function
def model_fn(batch, model, criterion, device):
    """Forward a batch through the model."""

    mels, labels = batch

    mels = mels.to(device)
    labels = labels.to(device)

    outs = model(mels)

    loss = criterion(outs, labels)

    # Get the speaker id with the highest probability.
    preds = outs.argmax(axis=1)

    # Compute accuracy.
    accuracy = torch.mean((preds == labels).float())

    return loss, accuracy


# Validate
def valid(dataloader, model, criterion, device):
    """Validate on validation set."""

    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()

        pbar.update(dataloader.batch_size)
        pbar.set_postfix(
            loss=f"{running_loss / (i + 1):.2f}",
            accuracy=f"{(running_accuracy / (i + 1)) * 100:.2f}%",
        )

    pbar.close()
    model.train()

    return running_accuracy / len(dataloader)


# Main Function of Training
def parse_args():
    """arguments"""
    config = {
        "data_dir": "../dataset",
        "save_path": "../checkpoints/model.ckpt",
        "batch_size": 32,
        "n_workers": 8,
        "valid_steps": 2000,
        "warmup_steps": 1000,
        "save_steps": 10000,
        "total_steps": 70000,
        "num_epochs": 10,
    }
    return config


def main(
        data_dir,
        save_path,
        batch_size,
        n_workers,
        valid_steps,
        warmup_steps,
        total_steps,
        save_steps,
        num_epochs: int = 10,
):
    """Main function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size, n_workers)
    # train_iterator = iter(train_loader)
    print(f"[Info]: Finish loading data!", flush=True)

    model = Classifier(n_spks=speaker_num)
    if os.path.exists(save_path):
        ckpt_weights = torch.load(save_path)
        model.load_state_dict(ckpt_weights)
        print(f"[Info]: Checkpoint loaded!", flush=True)
    else:
        print(f"[Info]: No checkpoint found, model randomly initialized!", flush=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[Info]: Finish creating model!", flush=True)

    best_accuracy = -1.0
    best_state_dict = None

    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    viz = Visdom()
    viz.line([0.], [0.], win="train_loss", opts=dict(title="train_loss"))

    step = 0
    for epoch in range(num_epochs):
        for batch in train_loader:
            loss, accuracy = model_fn(batch, model, criterion, device)
            batch_loss = loss.item()
            batch_accuracy = accuracy.item()

            # Update model
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Log
            viz.line([batch_loss], [step], win="train_loss", update="append")
            pbar.update()
            pbar.set_postfix(
                loss=f"{batch_loss:.2f}",
                accuracy=f"{batch_accuracy * 100:.2f}%",
                step=step + 1,
            )

            # Do validation
            if (step + 1) % valid_steps == 0:  # 每valid_steps次训练后，进行一次validation
                pbar.close()

                valid_accuracy = valid(valid_loader, model, criterion, device)

                # keep the best model
                if valid_accuracy > best_accuracy:
                    best_accuracy = valid_accuracy
                    best_state_dict = model.state_dict()

                pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

            # Save the best model so far.
            if (step + 1) % save_steps == 0 and best_state_dict is not None:
                torch.save(best_state_dict, save_path)
                pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy * 100:.2f}%)")

            # 更新训练阶段的全局参数step
            step += 1

    pbar.close()


if __name__ == "__main__":
    main(**parse_args())
