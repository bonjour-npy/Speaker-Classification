# Speaker Classification

![image-20240113175506430](https://raw.githubusercontent.com/bonjour-npy/Image-Hosting-Service/main/typora_imagesimage-20240113175506430.png)

## Overview

Classify the speaker of given features, learn how to use Transformer and how to adjust parameters of transformer.

## Dataset

The original dataset is [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/).

We randomly select 600 speakers from Voxceleb1, then preprocess the raw waveforms into mel-spectrograms.

![image-20240113163045453](https://raw.githubusercontent.com/bonjour-npy/Image-Hosting-Service/main/typora_imagesimage-20240113163045453.png)

Args:

- data_dir: The path to the data directory.

- metadata_path: The path to the metadata.

- segment_len: The length of audio segment for training.

The architecture of dataset directory:

```
data directory/
├── mapping.json
├── metadata.json
└── testdata.json
```

## Related

This is also the assignment solution of [ML2021Spring HW4](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.php).