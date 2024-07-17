import argparse

import onnxruntime as rt
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--dir_onnx", type=str, help="Path to the onnx model")
args = parser.parse_args()

dir_onnx = args.dir_onnx
if "layoutlmv3-base" in dir_onnx:
    has_visual_embeds = True

session = rt.InferenceSession(
    dir_onnx,
    providers=["CPUExecutionProvider"],
)


# backbone args
input_ids = torch.randint(1, 200000, (1, 512)).numpy()
bbox = torch.zeros((1, 512, 4)).long().numpy()
attention_mask = torch.ones((1, 512)).long().numpy()

# peneo decoder args
orig_bbox = torch.zeros((1, 512, 4)).long().numpy()

inputs = {
    "input_ids": input_ids,
    "bbox": bbox,
    "orig_bbox": orig_bbox,
    "attention_mask": attention_mask,
}

# image_input
if has_visual_embeds:
    input_image = torch.rand((1, 3, 224, 224), dtype=torch.float32).numpy()
    inputs.update({"image": input_image})

outputs = session.run([], inputs)
print("End")
