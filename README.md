## Overview
FedSDAF is a novel federated learning framework addressing domain shift challenges in non-IID data scenarios.
<div align="center">
    <img src="figure/Optimization_process.png" alt="Optimization Process" width="50%">
</div>

## Installation
Create and activate conda environment
```sh
conda create -n FedSDAF python=3.9.20
conda activate FedSDAF
pip install -r requirements.txt
```

## Datasets
[PACS](https://domaingeneralization.github.io/#data)

[OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html)

[VLCS](https://github.com/belaalb/G2DM#download-vlcs)

[DomainNet](https://ai.bu.edu/M3SDA/)

Please put the datasets in the following directory structure:

```
├── datasets
│   ├── pacs
│   │   ├── raw_images
│   │   │   ├── art_painting
│   │   │   ├── cartoon
│   │   │   ├── photo
│   │   │   ├── sketch
│   │   ├── Train val splits and h5py files pre-read
│   ├── office_home
│   │   ├── Art
│   │   ├── Clipart
│   │   ├── Product
│   │   ├── Real_World
│   ├── vlcs
│   │   ├── Caltech101
│   │   ├── LabelMe
│   │   ├── SUN09
│   │   ├── VOC2007
│   ├── domain_net
│   │   ├── clipart
│   │   ├── infograph
│   │   ├── painting
│   │   ├── quickdraw
│   │   ├── real
│   │   ├── sketch
```

Then set the root directory in the make_datasets_path.sh file and run:

```sh
sh make_datasets_path.sh
```

## Train
Please set the root directory in the train.sh file. Then run:

```sh
sh train.sh
```

The log and checkpoints files will be saved in ./FedSDAF/log.

## Acknowledgments 

Part of our code is borrowed from the repositories [FedDG-GA](https://github.com/MediaBrain-SJTU/FedDG-GA) and [PLAN](https://github.com/GongShuai8210/PLAN). We thank them for sharing the code.