---
base_model: google/gemma-2-2b-it
library_name: transformers
model_name: func_call_gemma-2-2B-V0
tags:
- generated_from_trainer
- trl
- sft
licence: license
---

# Model Card for func_call_gemma-2-2B-V0

This model is a fine-tuned version of [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="ishikakulkarni/func_call_gemma-2-2B-V0", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- TRL: 0.16.0
- Transformers: 4.50.2
- Pytorch: 2.6.0
- Datasets: 3.5.0
- Tokenizers: 0.21.1
