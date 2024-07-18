<h1>PEneo</h1>

This is an official re-implementation of PEneo introduced in the paper *PEneo: Unifying Line Extraction, Line Grouping, and Entity Linking for End-to-end Document Pair Extraction*. The RFUND annotations proposed in this paper can be found at [SCUT-DLVCLab/RFUND](https://github.com/SCUT-DLVCLab/RFUND).

> Codes in this repository have undergone modifications from our original implementation to enhance its flexibility and usability. As a result, the model performance may vary slightly from the original implementation.


<h2>Table of Contents</h2>

- [Setup](#setup)
  - [Installation](#installation)
  - [Dataset Preparation](#dataset-preparation)
  - [Backbone Preparation](#backbone-preparation)
    - [Supported Document-AI backbones](#supported-document-ai-backbones)
    - [Pre-trained Utils Generation](#pre-trained-utils-generation)
- [Fine-tuning](#fine-tuning)
  - [Pair Extraction on RFUND](#pair-extraction-on-rfund)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [Copyright](#copyright)


## Setup

### Installation

```bash
conda create -n vie python=3.10
conda activate vie
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
#pip install -r requirements.txt
```

If you want to use LayoutLMv2/LayoutXLM backbone, please additionally install detectron2:

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```


### Dataset Preparation

The RFUND annotations can be downloaded from [here](https://github.com/SCUT-DLVCLab/RFUND). Images of the dataset is available at the original release of [FUNSD](https://guillaumejaume.github.io/FUNSD/) and [XFUND](https://github.com/doc-analysis/XFUND/releases/tag/v1.0). The downloaded dataset should be organized as follows:

```
private_data
└── rfund
    ├── images
    │   ├── de
    │   ├── en
    │   ├── es
    │   ├── fr
    │   ├── it
    │   ├── ja
    │   ├── pt
    │   └── zh
    ├── de.train.json
    ├── de.val.json
    ├── en.train.json
    ├── en.val.json
    ├── es.train.json
    ├── es.val.json
    ├── fr.train.json
    ├── fr.val.json
    ├── it.train.json
    ├── it.val.json
    ├── ja.train.json
    ├── ja.val.json
    ├── pt.train.json
    ├── pt.val.json
    ├── zh.train.json
    └── zh.val.json
```

### Backbone Preparation

#### Supported Document-AI backbones

| Model Name              | 🤗 Link                                                                                        |
| ----------------------- | --------------------------------------------------------------------------------------------- |
| lilt-infoxlm-base       | [SCUT-DLVCLab/lilt-infoxlm-base](https://huggingface.co/SCUT-DLVCLab/lilt-infoxlm-base)       |
| lilt-roberta-en-base    | [SCUT-DLVCLab/lilt-roberta-en-base](https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base) |
| layoutxlm-base          | [microsoft/layoutxlm-base](https://huggingface.co/microsoft/layoutxlm-base)                   |
| layoutlmv2-base         | [microsoft/layoutlmv2-base-uncased](https://huggingface.co/microsoft/layoutlmv2-base-uncased) |
| layoutlmv3-base         | [microsoft/layoutlmv3-base](https://huggingface.co/microsoft/layoutlmv3-base)                 |
| layoutlmv3-base-chinese | [microsoft/layoutlmv3-base-chinese](https://huggingface.co/microsoft/layoutlmv3-base-chinese) |


#### Pre-trained Utils Generation

If you want to use layoutlmv3-base as the model backbone, you can generate the required files by running the following command:

```bash
python tools/generate_peneo_weights.py --backbone_name_or_path microsoft/layoutlmv3-base --output_dir private_pretrained/layoutlmv3-base
```

The scripts will automatically download the pre-trained weights, tokenizer, and config files from 🤗Huggingface hub and convert them to the required format. If you want to use other backbones, you can change the `--backbone_name_or_path` parameter to the corresponding HF model id.

If the scripts failed to download the pre-trained files, you may manually download them through the links in the above table, and set the `--backbone_name_or_path` parameter to the local directory of the downloaded files.


## Fine-tuning

Checkpoints, terminal outputs, and tensorboard logs will be saved in `private_output/weights`, `private_output/logs`, and `private_output/runs`, respectively. Please create these directories before running the scripts.

```bash
mkdir -p private_output/weights
mkdir -p private_output/logs
mkdir -p private_output/runs
```

### Pair Extraction on RFUND

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0,1
export TRANSFORMERS_NO_ADVISORY_WARNINGS='true'
PROC_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")
MASTER_PORT=11451

LANGUAGE=en
TASK_NAME=layoutlmv3-base_rfund_${LANGUAGE}
PRETRAINED_PATH=private_pretrained/layoutlmv3-base
DATA_DIR=private_data/rfund
OUTPUT_DIR=private_output/weights/$TASK_NAME
RUNS_DIR=private_output/runs/$TASK_NAME
LOG_DIR=private_output/logs/$TASK_NAME.log
torchrun --nproc_per_node $PROC_PER_NODE --master_port $MASTER_PORT start/run_rfund.py \
    --model_name_or_path $PRETRAINED_PATH \
    --data_dir $DATA_DIR \
    --language $LANGUAGE \
    --output_dir $OUTPUT_DIR \
    --do_train \
    --do_eval \
    --fp16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --dataloader_num_workers 8 \
    --warmup_ratio 0.1 \
    --learning_rate 5e-5 \
    --max_steps 25000 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --logging_strategy epoch \
    --logging_dir $RUNS_DIR \
    --detail_eval True \
    --save_eval_detail True \
    2>&1 | tee -a $LOG_DIR
```

The above script use `layoutlmv3-base` as the model backbone and fine-tune the model on `RFUND-EN`. You may try different backbones and language subsets by changing the `PRETRAINED_PATH` and `LANGUAGE` accordingly.


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

## Acknowledgement

Part of the code is adapted from [LiLT](), [LayoutLMv2/XLM](https://github.com/microsoft/unilm/tree/master/layoutlmft), [LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3), and [TPLinker](https://github.com/131250208/TPlinker-joint-extraction). We sincerely thank the authors for their great work.


## Copyright

This repository can only be used for non-commercial research purposes. For other purposes, please contact Prof. Lianwen Jin (eelwjin@scut.edu.cn).

Copyright 2024, [Deep Learning and Vision Computing Lab](http://www.dlvc-lab.net/), South China University of Technology.