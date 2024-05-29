# PEneo

This is an official re-implementation of PEneo introduced in the paper *PEneo: Unifying Line Extraction, Line Grouping, and Entity Linking for End-to-end Document Pair Extraction*.

> Codes in this repository have undergone modifications from our original implementation to enhance its usability and comprehensibility. As a result, slight performance variations on the benchmarks may be observed.


# Environment Setup

## Use conda & pip

```bash
conda create -n vie python=3.10
conda activate vie
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
```

If you want to use layoutlmv2/layoutxlm backbone, please install detectron2 additionally:

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Use Docker

### Build Docker Image Locally

```bash
docker build -t peneo:v1.0 .
```

If you want to use layoutlmv2/layoutxlm backbone, please install detectron2 inside the docker container additionally



## Citation

If you find PEneo helpful, please consider citing our paper:

```
@article{lin2024peneo,
  title={PEneo: Unifying Line Extraction, Line Grouping, and Entity Linking for End-to-end Document Pair Extraction},
  author={Lin, Zening and Wang, Jiapeng and Li, Teng and Liao, Wenhui and Huang, Dayi and Xiong, Longfei and Jin, Lianwen},
  journal={arXiv preprint arXiv:2401.03472},
  year={2024}
}
```

## Copyright

This repository can only be used for non-commercial research purposes. For other purposes, please contact Prof. Lianwen Jin (eelwjin@scut.edu.cn).

Copyright 2024, [Deep Learning and Vision Computing Lab](http://www.dlvc-lab.net/), South China University of Technology.