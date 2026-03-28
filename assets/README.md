# Watermark Detection Instructions

This model is watermarked using OpenStamp. To detect watermarked text, use the detection utilities in `src.openstamp`.

## Quickstart

1. Clone this repository and install dependencies as described in the main project README.
2. Ensure you have the selector matrix and watermark config files (these are included in this model repo).
3. Use the detection script or import the detection functions from `src/mbmark.py`.

Example usage:

```python
from src.openstamp import MbMark, Mode
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained('.')
tokenizer = AutoTokenizer.from_pretrained('.')

# Load selector matrix and config
selector_matrix = torch.load('selector_matrix.pth', map_location='cpu')
with open('watermark_config.json') as f:
    config = json.load(f)

# Initialize detector
mb = MbMark.mb(
    delta=config['delta'],
    gamma=config['gamma'],
    seed=config['seed'],
    final_weight=selector_matrix,
    model=model,
    tokenizer=tokenizer,
    unembedding_param_name='lm_head',
    mode=Mode.Detect,
)

# Detect watermark in text
text = "your generated text here"
inputs = tokenizer(text, return_tensors='pt')
with torch.no_grad():
    result = mb.detect(inputs['input_ids'])
print(result)
```

- For more details, see the main project documentation.
- For custom detection or batch processing, adapt the above code as needed.
