<h1> 🚀 PEneo Documentation </h1>

- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
  - [Use Existing Academic Datasets](#use-existing-academic-datasets)
    - [RFUND](#rfund)
    - [SIBR](#sibr)
  - [Adapt to Custom Datasets](#adapt-to-custom-datasets)
- [Backbone Preparation](#backbone-preparation)
  - [Supported Document-AI backbones](#supported-document-ai-backbones)
  - [Pre-trained Utils Generation](#pre-trained-utils-generation)
- [Model Configurations](#model-configurations)
- [Fine-tuning](#fine-tuning)
- [Deployment](#deployment)
  - [Export the ONNX model](#export-the-onnx-model)
  - [Inference with PyTorch weights](#inference-with-pytorch-weights)
  - [Inference with ONNX model](#inference-with-onnx-model)


## Repository Structure

```bash
PEneo
├── data
│   ├── collator.py             # Data collator
│   ├── datasets                # Dataset pre-processing pipeline
│   └── data_utils.py           # Data processing utilities
├── deploy                      # Inference scripts
├── docs                        # Documentation
├── model                       # Model architecture and configuration
│   ├── backbone                # Implementation of the backbone models
│   ├── backbone_mapping.py     # Mappings for the backbone models
│   ├── configuration_peneo.py  # HF style Model configuration
│   ├── custom_loss.py          # Custom loss functions including OHEM
│   ├── modeling_peneo.py       # PEneo model implementation
│   └──  peneo_decoder.py       # PEneo downstream head implementation
├── pipeline
│   ├── decode.py               # Decode the model output, generate the kv-pairs
│   ├── evaluation.py           # Metrics calculation
│   └── trainer.py              # HF Trainer implementation
├── private_data                # Directory to store the dataset
├── private_output              # Directory to store the model weights and logs
├── private_pretrained          # Directory to store the pre-trained model
├── start                       # Training scripts
└── tools
    ├── check_run_onnx.py       # Check the onnx model output
    ├── export_onnx.py          # Export the onnx model
    └── generate_peneo_weights.py  # Generate the pre-trained model utils
```

The `private_data` directory is used to store the dataset. It can be organized as follows:

```bash
private_data
    ├── rfund -> /real/path/to/RFUND
    └── sibr -> /real/path/to/XFUND
```

The `private_output` directory is used to store the model weights and logs. It should be organized as follows:

```bash
private_output
    ├── runs # Directory to store the tensorboard logs
    ├── logs # Directory to store the terminal outputs
    └── weights # Directory to store the model weights
```
 
The `private_pretrained` directory is used to store the pre-trained model weights. It may be organized as follows:

```bash
private_pretrained
    ├── layoutlmv2-base-uncased
    ├── layoutlmv3-base
    ├── layoutlmv3-base-chinese
    ├── layoutxlm-base
    └── lilt-infoxlm-base

```

These three folders should be created manually when you start the project. Contents in these directories will not be traced by git.


## Installation

```bash
conda create -n vie python=3.10
conda activate vie
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
```

If you want to use LayoutLMv2/LayoutXLM backbone, please additionally install detectron2:

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```


## Dataset Preparation

### Use Existing Academic Datasets

#### RFUND

The RFUND annotations can be downloaded from [here](https://github.com/SCUT-DLVCLab/RFUND). Images of the dataset is available at the original release of [FUNSD](https://guillaumejaume.github.io/FUNSD/) and [XFUND](https://github.com/doc-analysis/XFUND/releases/tag/v1.0). The downloaded dataset should be organized as follows:

```bash
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

#### SIBR

We notice that some annotation errors exist in the original SIBR dataset (mainly due to the failure of data masking rules). To avoid potential issues, we made manual corrections and made the revised labels available [here](https://github.com/ZeningLin/PEneo/releases/tag/SIBR-revised-v1.0). Images of the dataset are available at the original release of [SIBR](https://www.modelscope.cn/datasets/iic/SIBR).

After downloading the original SIBR dataset and our revised labels, you should extract and place the revised `converted_label` folder under the root of the original SIBR directory. The dataset should be organized as follows:

```bash
private_data
└── sibr
    ├── converted_label # revised labels
    ├── images          # original images
    ├── label           # original labels
    ├── train.txt       # train split file
    └── test.txt        # test split file
```

### Adapt to Custom Datasets

You can refer to the dataloader of the SIBR dataset in `data/datasets/sibr.py` to construct the dataloader for your custom dataset. The `__getitem__` method implement the following processing steps:

- Load a sample image and its corresponding annotation file
- Iterate over the entity annotations, collect the line information within the entity, including:
  - Line bounding box
  - Line tokens split by the tokenizer
  - The original texts that each token correspond to. In the post processing step, we need to restore the text content of each key/value pairs from the token-level outputs. Since the tokenizer may remove or add special tokens, we need to align the tokenized text with the original text. We use the `tokenizer_fetcher` implemented in `model/backbone_mapping.py` to fetch the original text.
- Sort the lines by coordinates from left-top-right-bottom order.
- Generate the list of line token ids, normalized bounding boxes (range from 0 to 1000), original bounding boxes, and the original texts.
- According to the start and end token index of each key/value lines, generate the label `line_extraction_matrix_spots` for the line extraction task. The label is a list containing tuples in the format of (line_start_token_idx, line_end_token_idx, 1). The last term 1 indicates that the label is positive. For negative lines, they are not required to be included in the label list.
- According to the key-value linking annotations, generate the label `ent_linking_head_rel_matrix_spots` and `ent_linking_tail_rel_matrix_spots` for the entity linking task. The label is a list containing tuples in the format of (key_first_line_start_token_idx, value_first_line_start_token_idx, label_type). To reduce computational cost in the downstream pairwise matrix, we flip up the content in the lower triangle to the upper part. For example, if the original label is (10, 2), then it will be flipped to (2, 10) in the label list, and its label_type is set to 2 to indicate the flip operation. For other labels, the label_type is set to 1.
- According to the line grouping annotations (neighboring relation of lines within an entity), generate the label `line_grouping_head_rel_matrix_spots` and `line_grouping_tail_rel_matrix_spots` for the line grouping task. The label is a list containing tuples in the format of (prev_line_start_token_idx, next_line_start_token_idx, label_type). Similar to the entity linking task, we flip up the content in the lower triangle to the upper part and use the label_type to indicate the flip operation.
- For reference, we also generate a `relations` term that contains the key-value pair texts. This term is currently not used in the training/evaluation process but can be used for debugging purposes.

Finally, the dataloader will return the following terms:

- `fname`: The file name of the sample
- `image_path`: The full path of the image, will be used in the data collate function to load the image if required by the backbone model.
- `input_ids`: List of token ids in the document.
- `bbox`: List of normalized bounding boxes of the lines. With the same length as the input_ids.
- `original_bbox`: List of original bounding boxes of the lines. With the same length as the input_ids.
- `text`: List of original texts that each token corresponds to. With the same length as the input_ids.
- `relations`: List of dict containing key-value pair texts.
- `line_extraction_matrix_spots`: List of tuples in the format of (line_start_token_idx, line_end_token_idx, 1) for the line extraction task.
- `ent_linking_head_rel_matrix_spots`: List of tuples in the format of (key_first_line_start_token_idx, value_first_line_start_token_idx, label_type) for the head linking subtask in entity linking.
- `ent_linking_tail_rel_matrix_spots`: List of tuples in the format of (key_first_line_start_token_idx, value_first_line_start_token_idx, label_type) for the tail linking subtask in entity linking.
- `line_grouping_head_rel_matrix_spots`: List of tuples in the format of (prev_line_start_token_idx, next_line_start_token_idx, label_type) for the head linking subtask in line grouping.
- `line_grouping_tail_rel_matrix_spots`: List of tuples in the format of (prev_line_start_token_idx, next_line_start_token_idx, label_type) for the tail linking subtask in line grouping.


The following class objects are used in the data processing:

- `ENTITY_LABEL_LIST`: List of entity types. Modify based on your custom dataset. Remember to keep the background label at the first position.
- `LABEL_LIST`: List of entity types in BIO format. Modify based on your custom dataset. Remember to keep the background label "O" at the first position.
- `LABEL_NAME2ID` and `LABEL_ID2NAME`: Dictionaries that map the entity type to its corresponding ID and vice versa.


You can refer to the [SIBR annotations](https://github.com/ZeningLin/PEneo/releases) released in this repository and convert the annotation of your custom dataset to the same format. Then you can construct your own dataloader based on the `SIBRDataset` with slight modifications. The SIBR formats are as follows:

```json
{
    "uid": "str <sample_id>",
    "img": {
        "fname": "str <image_file_name>",
        "width": "int <image_width>",
        "height": "int <image_height>"
    },
    "entities": [
        {
            "id": "int <entity_id>",
            "label": "str <entity_type>",
            "lines": [
                {
                    "id": "int or str <line_id>",
                    "text": "str <line_text>",
                    "bbox": [
                        "int <left>",
                        "int <top>",
                        "int <right>",
                        "int <bottom>"
                    ]
                },
                ...
            ]
        },
        ...
    ],
    "relations": {
        "kv_entity": [
            {
                "from_id": "int <key_entity_id>",
                "to_id": "int <value_entity_id>"
            },
            ...
        ],
        "line_grouping": [
            {
                "from_id": "int <prev_line_id>",
                "to_id": "int <next_line_id>"
            },
            ...
        ]
    }
}

```

## Backbone Preparation

### Supported Document-AI backbones

<div align="center">

| Model Name              | Link                                                                                            |
| ----------------------- | ----------------------------------------------------------------------------------------------- |
| lilt-infoxlm-base       | 🤗 [SCUT-DLVCLab/lilt-infoxlm-base](https://huggingface.co/SCUT-DLVCLab/lilt-infoxlm-base)       |
| lilt-roberta-en-base    | 🤗 [SCUT-DLVCLab/lilt-roberta-en-base](https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base) |
| layoutxlm-base          | 🤗 [microsoft/layoutxlm-base](https://huggingface.co/microsoft/layoutxlm-base)                   |
| layoutlmv2-base-uncased | 🤗 [microsoft/layoutlmv2-base-uncased](https://huggingface.co/microsoft/layoutlmv2-base-uncased) |
| layoutlmv3-base         | 🤗 [microsoft/layoutlmv3-base](https://huggingface.co/microsoft/layoutlmv3-base)                 |
| layoutlmv3-base-chinese | 🤗 [microsoft/layoutlmv3-base-chinese](https://huggingface.co/microsoft/layoutlmv3-base-chinese) |

</div>

### Pre-trained Utils Generation

The pre-trained contents will be stored in the `private_pretrained` directory. Please create this folder before running the utils-generation scripts.

```bash
mkdir private_pretrained
```

If you want to use layoutlmv3-base as the model backbone, you can generate the required files by running the following command:


```bash
python tools/generate_peneo_weights.py \
  --backbone_name_or_path microsoft/layoutlmv3-base \
  --output_dir private_pretrained/layoutlmv3-base
```

The scripts will automatically download the pre-trained weights, tokenizer, and config files from 🤗Huggingface hub and convert them to the required format. Results will be stored in the `private_pretrained` directory. If you want to use other backbones, you can change the `--backbone_name_or_path` parameter to the corresponding HF model ID.

If the scripts fail to download the pre-trained files, you may manually download them through the links in the above table, and set the `--backbone_name_or_path` parameter to the local directory of the downloaded files.


## Model Configurations

When initializing the model, the `transformers` library will load the `config.json` in the pre-trained model directory and construct a `PEneoConfig` object to control the model's architecture. You can find the parameters in `model/configuration_peneo.py`:

- `backbone_name`: The name of the backbone model to use. currently supports
  -  lilt-infoxlm-base
  -  lilt-roberta-en-base
  -  layoutxlm-base
  -  layoutlmv2-base-uncased
  -  layoutlmv3-base-chinese
  -  layoutlmv3-base
- `backbone_config`: The huggingface transformers configuration for the backbone model. Will automatically download and integrate from the huggingface model hub when generating the pre-trained utils. No need to modify.
- `initializer_range`: The standard deviation of the normalized weights in the downstream layers.
- `peneo_decoder_shrink`: Whether to reduce the hidden size of the backbone output features to half. Default to True to reduce computational cost.
- `peneo_classifier_num_layers`: The number of linear layers in the five matrix classifiers.
- `peneo_loss_ratio`: The loss ratio of the five matrix classifiers. The loss of each classifier will be multiplied by this ratio.
- `peneo_category_weight`: The loss weight of each category in the cross-entropy loss. In our experiments, we set the weight of the background category to 1 and the weight of the other categories to 10.
- `peneo_ohem_num_positive`: The number of positive samples to keep in the online hard example mining process. OHEM will be activated when the value is greater than 0. Default to -1 to disable OHEM.
- `peneo_ohem_num_negative`: The number of negative samples to keep in the online hard example mining process. OHEM will be activated when the value is greater than 0. Default to -1 to disable OHEM.
- `peneo_downstream_speedup_ratio`: The learning rate of the downstream layers will be multiplied by this ratio. Default to 1 to keep the same learning rate. In our experiments, we set this value to 30. You can adjust this value based on your custom dataset.
- `inference_mode`: Set to True only when exporting the onnx model to fit the tracing process. Default to False.


## Fine-tuning

You can fine-tune the model using the following command:

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0,1
export TRANSFORMERS_NO_ADVISORY_WARNINGS='true'
PROC_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")
PORT=11451

TASK_NAME=layoutlmv3_rfund_1                         # Task name, will be used as the directory name to save the model weights and logs
PRETRAINED_PATH=private_pretrained/layoutlmv3-base  # Pre-trained model path
BOX_AUG=False                                       # Whether to use box augmentation
DATA_DIR=private_data/rfund                         # Dataset path
OUTPUT_DIR=private_output/weights/$TASK_NAME        # Output directory
LOG_DIR=private_output/logs/$TASK_NAME              # Terminal Log directory
RUNS_DIR=private_output/runs/$TASK_NAME             # Tensorboard log directory
torchrun --nproc_per_node $PROC_PER_NODE --master_port $PORT start/run_rfund.py \
    --model_name_or_path $PRETRAINED_PATH \
    --data_dir $DATA_DIR \
    --language $LANGUAGE \
    --apply_box_aug $BOX_AUG \
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
    --save_total_limit 3 \
    --logging_strategy epoch \
    --logging_dir $RUNS_DIR \
    --detail_eval True \
    --save_eval_detail True \
    2>&1 | tee -a $LOG_DIR
```

You can monitor the training process through tensorboard:

```bash
tensorboard --logdir private_output/runs
```

The model weights will be saved in the `private_output/weights` directory.


## Deployment

### Export the ONNX model

You can export the model to ONNX format using the following command:

```bash
TASK_NAME=layoutlmv3_rfund_1  # Modify to the task name you want to export

python tools/export_onnx.py \
  --model_name_or_path private_output/weights/$TASK_NAME \
  --output_path private_output/weights/$TASK_NAME/peneo.onnx
```

> It is reported that some configurations and weights of the tokenizer will not be automatically saved in the private_output/weights by the huggingface trainer. Before exporting the onnx model, you may need to manually check and copy the missing files like `tokenizer_config.json`, `special_tokens_map.json`, `tokenizer.json`, `vocab.txt`, `sentencepiece.bpe.model`, etc. to the model directory.


To validate whether the exported ONNX model is bug-free, you can run the following command:

```bash
TASK_NAME=layoutlmv3_rfund_1  # Modify to the task name you want to validate

python tools/check_run_onnx.py \
  --dir_onnx private_output/weights/$TASK_NAME/peneo.onnx
```

### Inference with PyTorch weights

```bash
TASK_NAME=layoutlmv3_rfund_1  # Modify to the task name you want to validate

python deploy/inference.py \
  --model_name_or_path private_output/weights/$TASK_NAME \
  --dir_image /path/to/your/image \
  --dir_ocr /path/to/the/image/ocr/result \
  --visualize_path /path/to/save/the/visualization
```

The OCR results should be prepared in the following format:
```json
[
    {
        "text": "<str line_text_content>", 
        "bbox": [
            "int <left>",
            "int <top>",
            "int <right>",
            "int <bottom>"
        ]
    },
    ...
]
```

If you don't have the OCR results, you can use the huggingface built-in Tesseract OCR engine. You need to additionally install Tesseract OCR engine and the pytesseract package:

```bash
sudo apt install tesseract-ocr
pip install pytesseract
```

Then you can run the following command:

```bash
TASK_NAME=layoutlmv3_rfund_1  # Modify to the task name you want to validate

python deploy/inference.py \
  --model_name_or_path private_output/weights/$TASK_NAME \
  --dir_image /path/to/your/image \
  --visualize_path /path/to/save/the/visualization \
  --apply_ocr True
```

It is worth noting that the OCR results generated by the Tesseract OCR engine includes some special tokens like "ú", "í", etc., which may lead to failure in the token to original text mapping process in the tokenizer_fetcher. You may need to modify the `_special_text_replace` function in `deploy/inference.py/InferenceService` accordingly to handle these cases.


### Inference with ONNX model

```bash
TASK_NAME=layoutlmv3_rfund_1  # Modify to the task name you want to validate

python deploy/inference_onnx.py \
  --model_name_or_path private_output/weights/$TASK_NAME/peneo.onnx \
  --dir_image /path/to/your/image \
  --dir_ocr /path/to/the/image/ocr/result \
  --visualize_path /path/to/save/the/visualization
```

or using the built-in PyTesseract OCR engine:

```bash
TASK_NAME=layoutlmv3_rfund_1  # Modify to the task name you want to validate

python deploy/inference_onnx.py \
  --model_name_or_path private_output/weights/$TASK_NAME/peneo.onnx \
  --dir_image /path/to/your/image \
  --visualize_path /path/to/save/the/visualization \
  --apply_ocr True
```

