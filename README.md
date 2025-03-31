# Function Calling with Transparent Reasoning

Fine-tuned version of Gemma-2-2B to perform structured function calls with explicit reasoning steps. This model demonstrates transparent thinking processes before making function calls.

## Key Features

1. **Transparent Reasoning**: The model exposes its reasoning process within `<think>...</think>` tags, making its decision-making visible and auditable.

2. **Structured Function Calling**: Implements a standardized format for function calls using `<tool_call>...</tool_call>` tags containing properly formatted JSON objects that comply with provided schemas.

3. **Parameter Extraction**: The model correctly identifies and extracts parameters from user queries, parsing them into the appropriate data types for function arguments.

4. **Multiple Tool Support**: Handles selection between multiple available tools based on user intent, choosing the most appropriate function for each situation.

5. **Format Consistency**: Maintains consistent output formatting, making it reliable for integration with downstream applications that expect standardized function calling formats.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "ishikakulkarni/func_call_gemma-2-2B-V0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = """<bos><start_of_turn>human
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.
Here are the available tools:
<tools> [{'type': 'function', 'function': {'name': 'convert_currency', 'description': 'Convert from one currency to another', 'parameters': {'type': 'object', 'properties': {'amount': {'type': 'number', 'description': 'The amount to convert'}, 'from_currency': {'type': 'string', 'description': 'The currency to convert from'}, 'to_currency': {'type': 'string', 'description': 'The currency to convert to'}}, 'required': ['amount', 'from_currency', 'to_currency']}}}] </tools>

Use the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{tool_call}
</tool_call>

Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>

I need to convert 50 USD to EUR.<end_of_turn><eos>
<start_of_turn>model"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=500)
result = tokenizer.decode(outputs[0])
print(result)
