import os
import numpy as np
import torch
from torch.amp import autocast
import onnxruntime as ort

from model import Att3DUNet
from hyperparameters import (
    IN_CH, OUT_CH, NUM_FILTERS, NUM_HEADS
)

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)

# hyperparameters
in_ch = IN_CH
out_ch = OUT_CH
num_filters = NUM_FILTERS
num_heads = NUM_HEADS

# model setup
model = Att3DUNet(
    in_channels=in_ch, out_channels=out_ch, num_filters=num_filters, num_heads=num_heads, dropout=False
)

model.to(device)

model.load_state_dict(torch.load('weights/best_weights.pt', map_location=device))

model.eval()

# make ONNX directory
os.makedirs('onnx', exist_ok=True)

# export model to ONNX using a dummy input
dummy_input = torch.randn(1, in_ch, 64, 64, 64, device=device)

with torch.no_grad():
    torch.onnx.export(
        model, (dummy_input,), 'onnx/brainseg_dynamo.onnx',
        input_names=['input'], output_names=['output'],
        opset_version=21, dynamo=True,
    )

# numerical parity check
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

onnx_path = 'onnx/brainseg_dynamo.onnx'

inf_sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
ort_out = inf_sess.run(['output'], {'input': dummy_input.cpu().numpy()})[0]

with torch.no_grad():
    torch_out = model(dummy_input).cpu().numpy()

max_diff = np.abs(torch_out - ort_out).max()
rel = max_diff / np.abs(ort_out).max()

# for segmentation we mainly want to see if the label map changes
agree = (torch_out.argmax(1) == ort_out.argmax(1)).mean()

print(f'\nparity vs onnxruntime cpu fp32:')
print(f'    max|diff| = {max_diff:.3e}, rel = {rel:.2e}, argmax agreement = {100 * agree:.2f}%')
assert agree == 1.0, 'ONNX export changes the predicted label map'

# comparing eager PyTorch vs. ONNX
def benchmark(fn, warmup=20, iterations=100):
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize() # work must finish before the clock is read
        times.append(start.elapsed_time(end))
    return np.array(times)

def report(name, t):
    print(f'    {name:<16} median {np.median(t):7.2f} ms   '
          f'p5 {np.percentile(t, 5):6.2f}   p95 {np.percentile(t, 95):6.2f}   '
          f'{1000 / np.median(t):5.1f} it/s')

def fp32_forward():
    model(dummy_input)

def fp16_forward():
    with autocast('cuda'):
        model(dummy_input)

print('\nbaseline (eager pytorch, no tf32):')
with torch.no_grad():
    report('fp32', benchmark(fp32_forward))
    report('fp16', benchmark(fp16_forward))
