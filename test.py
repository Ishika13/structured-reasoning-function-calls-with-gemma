from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import torch
import json
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Set device
cuda_device = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

username="ishikakulkarni"
output_dir = "func_call_gemma-2-2B-V0" 


# Set device maps and quantization configs
model_device_map = {"": cuda_device} 
print(f"Setting device map to: {model_device_map}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading tokenizer first")
tokenizer = AutoTokenizer.from_pretrained(f"{username}/{output_dir}")

def preprocess(sample):
    messages = sample["messages"]
    first_message = messages[0]

    if first_message["role"] == "system":
        system_message_content = first_message["content"]
        messages[1]["content"] = system_message_content + "Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>\n\n" + messages[1]["content"]
        messages.pop(0)

    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

# Load and preprocess dataset
print("Loading and Processing dataset")
dataset_name = "Jofthomas/hermes-function-calling-thinking-V1"
dataset = load_dataset(dataset_name)
dataset = dataset.rename_column("conversations", "messages")
dataset = dataset.map(preprocess, remove_columns="messages")
dataset = dataset["train"].train_test_split(0.1)
print(dataset["test"][8]["text"])

# Load finetuned model
print("Loading finetuned model config")
peft_model_id = f"{username}/{output_dir}"
config = PeftConfig.from_pretrained(peft_model_id)

# Load model
print("Loading base model")
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map=model_device_map, 
    quantization_config=bnb_config,
)
print("Resizing Token Embeddings")
model.resize_token_embeddings(len(tokenizer))

print("Loading PEFT adapter weights")
model = PeftModel.from_pretrained(model, peft_model_id)
model.eval() 

# Load base model for comparison with same device mapping
print("Loading base model for comparison")
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map=model_device_map,  
    quantization_config=bnb_config
)
base_model.resize_token_embeddings(len(tokenizer))
base_model.eval()

# Helper functions to extract tool calls and thinking
def extract_tool_call(text):
    tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    if tool_call_match:
        try:
            return json.loads(tool_call_match.group(1).strip())
        except:
            print("Failed to parse JSON from tool_call")
            print(f"Raw content: {tool_call_match.group(1).strip()}")
            return None
    return None

def extract_thinking(text):
    """Extract the thinking part from the model output"""
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    return None

# Function to evaluate a model on test cases
def evaluate_model(model, test_cases, name):
    results = []
    for i, test_case in enumerate(test_cases):
        print(f"\nEvaluating {name} on test case {i+1}/{len(test_cases)}")
        
        inputs = tokenizer(test_case["prompt"], return_tensors="pt")
        inputs = {k: v.to(f"cuda:{cuda_device}") for k, v in inputs.items()}
        
        try:
            outputs = model.generate(
                **inputs, 
                max_new_tokens=500, 
                do_sample=True,
                top_p=0.95,
                temperature=0.01,
                repetition_penalty=1.0,
                eos_token_id=tokenizer.eos_token_id
            )
            
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            tool_call = extract_tool_call(output_text)
            thinking = extract_thinking(output_text)
            
            # Evaluate results
            result = {
                "correct_function": False,
                "all_params_present": False,
                "correct_param_values": False,
                "has_thinking": thinking is not None,
            }
            
            if tool_call and "name" in tool_call:
                result["correct_function"] = tool_call["name"] == test_case["expected_function"]
                
                if "arguments" in tool_call:
                    # Check if all expected parameters are present
                    params_present = all(param in tool_call["arguments"] for param in test_case["expected_params"])
                    result["all_params_present"] = params_present
                    
                    # Check if parameter values match expected
                    if params_present:
                        correct_values = all(
                            str(test_case["expected_values"].get(param)) == str(tool_call["arguments"].get(param))
                            for param in test_case["expected_params"]
                        )
                        result["correct_param_values"] = correct_values
            
            results.append(result)
            
            # Print sample outputs for manual inspection
            if i < 2:  # Just show a couple examples
                print(f"\n{name} Output for Test Case {i+1}:")
                print(output_text[:500] + "..." if len(output_text) > 500 else output_text)
                print(f"Tool Call Extracted: {tool_call}")
                print(f"Thinking Extracted: {thinking}")
                print("-" * 50)
                
        except Exception as e:
            print(f"Error during generation for test case {i+1}: {str(e)}")
            # Add a failed result
            results.append({
                "correct_function": False,
                "all_params_present": False,
                "correct_param_values": False,
                "has_thinking": False,
                "error": str(e)
            })
    
    # Calculate metrics
    if not results:
        return {"function_accuracy": 0, "params_accuracy": 0, "values_accuracy": 0, "thinking_rate": 0}, []
    
    metrics = {
        "function_accuracy": sum(r["correct_function"] for r in results) / len(results),
        "params_accuracy": sum(r["all_params_present"] for r in results) / len(results),
        "values_accuracy": sum(r["correct_param_values"] for r in results) / len(results),
        "thinking_rate": sum(r["has_thinking"] for r in results) / len(results),
    }
    
    return metrics, results

# Define test cases
test_cases = [
    {
        "prompt": """<bos><start_of_turn>human
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.Here are the available tools:<tools> [{'type': 'function', 'function': {'name': 'convert_currency', 'description': 'Convert from one currency to another', 'parameters': {'type': 'object', 'properties': {'amount': {'type': 'number', 'description': 'The amount to convert'}, 'from_currency': {'type': 'string', 'description': 'The currency to convert from'}, 'to_currency': {'type': 'string', 'description': 'The currency to convert to'}}, 'required': ['amount', 'from_currency', 'to_currency']}}}, {'type': 'function', 'function': {'name': 'calculate_distance', 'description': 'Calculate the distance between two locations', 'parameters': {'type': 'object', 'properties': {'start_location': {'type': 'string', 'description': 'The starting location'}, 'end_location': {'type': 'string', 'description': 'The ending location'}}, 'required': ['start_location', 'end_location']}}}] </tools>Use the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{tool_call}
</tool_call>Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>

I need to convert 100 EUR to USD.<end_of_turn><eos>
<start_of_turn>model""",
        "expected_function": "convert_currency",
        "expected_params": ["amount", "from_currency", "to_currency"],
        "expected_values": {"amount": 100, "from_currency": "EUR", "to_currency": "USD"}
    },
    {
        "prompt": """<bos><start_of_turn>human
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.Here are the available tools:<tools> [{'type': 'function', 'function': {'name': 'convert_currency', 'description': 'Convert from one currency to another', 'parameters': {'type': 'object', 'properties': {'amount': {'type': 'number', 'description': 'The amount to convert'}, 'from_currency': {'type': 'string', 'description': 'The currency to convert from'}, 'to_currency': {'type': 'string', 'description': 'The currency to convert to'}}, 'required': ['amount', 'from_currency', 'to_currency']}}}, {'type': 'function', 'function': {'name': 'calculate_distance', 'description': 'Calculate the distance between two locations', 'parameters': {'type': 'object', 'properties': {'start_location': {'type': 'string', 'description': 'The starting location'}, 'end_location': {'type': 'string', 'description': 'The ending location'}}, 'required': ['start_location', 'end_location']}}}] </tools>Use the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{tool_call}
</tool_call>Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>

What's the distance from New York to Los Angeles?<end_of_turn><eos>
<start_of_turn>model""",
        "expected_function": "calculate_distance",
        "expected_params": ["start_location", "end_location"],
        "expected_values": {"start_location": "New York", "end_location": "Los Angeles"}
    },
    {
        "prompt": """<bos><start_of_turn>human
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.Here are the available tools:<tools> [{'type': 'function', 'function': {'name': 'convert_currency', 'description': 'Convert from one currency to another', 'parameters': {'type': 'object', 'properties': {'amount': {'type': 'number', 'description': 'The amount to convert'}, 'from_currency': {'type': 'string', 'description': 'The currency to convert from'}, 'to_currency': {'type': 'string', 'description': 'The currency to convert to'}}, 'required': ['amount', 'from_currency', 'to_currency']}}}, {'type': 'function', 'function': {'name': 'calculate_distance', 'description': 'Calculate the distance between two locations', 'parameters': {'type': 'object', 'properties': {'start_location': {'type': 'string', 'description': 'The starting location'}, 'end_location': {'type': 'string', 'description': 'The ending location'}}, 'required': ['start_location', 'end_location']}}}] </tools>Use the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{tool_call}
</tool_call>Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>

I want to convert 75 GBP to JPY.<end_of_turn><eos>
<start_of_turn>model""",
        "expected_function": "convert_currency",
        "expected_params": ["amount", "from_currency", "to_currency"],
        "expected_values": {"amount": 75, "from_currency": "GBP", "to_currency": "JPY"}
    },
]

# Evaluate single example first
print("Generating Output for Example Case")
prompt="""<bos><start_of_turn>human
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.Here are the available tools:<tools> [{'type': 'function', 'function': {'name': 'convert_currency', 'description': 'Convert from one currency to another', 'parameters': {'type': 'object', 'properties': {'amount': {'type': 'number', 'description': 'The amount to convert'}, 'from_currency': {'type': 'string', 'description': 'The currency to convert from'}, 'to_currency': {'type': 'string', 'description': 'The currency to convert to'}}, 'required': ['amount', 'from_currency', 'to_currency']}}}, {'type': 'function', 'function': {'name': 'calculate_distance', 'description': 'Calculate the distance between two locations', 'parameters': {'type': 'object', 'properties': {'start_location': {'type': 'string', 'description': 'The starting location'}, 'end_location': {'type': 'string', 'description': 'The ending location'}}, 'required': ['start_location', 'end_location']}}}] </tools>Use the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{tool_call}
</tool_call>Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>

Hi, I need to convert 500 USD to Euros. Can you help me with that?<end_of_turn><eos>
<start_of_turn>model"""

try:
    print("Generating Sample Output")
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(f"cuda:{cuda_device}") for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        top_p=0.95,
        temperature=0.01,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id
    )

    print("Decoding Sample Output")
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(output_text)

    print("\n----- Extracted Components -----")
    tool_call = extract_tool_call(output_text)
    thinking = extract_thinking(output_text)
    print(f"Tool Call: {tool_call}")
    print(f"Thinking Process: {thinking}")

    # Run comprehensive evaluation
    print("\n\n====== COMPREHENSIVE MODEL EVALUATION ======\n")
    print("Evaluating finetuned model...")
    ft_metrics, ft_results = evaluate_model(model, test_cases, "Finetuned Model")

    print("\nEvaluating base model...")
    base_metrics, base_results = evaluate_model(base_model, test_cases, "Base Model")

    # Print comparison
    print("\n----- Model Comparison Results -----")
    print(f"Function Accuracy: Base={base_metrics['function_accuracy']:.2%}, Finetuned={ft_metrics['function_accuracy']:.2%}")
    print(f"Parameters Accuracy: Base={base_metrics['params_accuracy']:.2%}, Finetuned={ft_metrics['params_accuracy']:.2%}")
    print(f"Values Accuracy: Base={base_metrics['values_accuracy']:.2%}, Finetuned={ft_metrics['values_accuracy']:.2%}")
    print(f"Thinking Rate: Base={base_metrics['thinking_rate']:.2%}, Finetuned={ft_metrics['thinking_rate']:.2%}")

    # Evaluate on test dataset samples
    def evaluate_on_test_dataset(model, test_dataset, num_samples=5): 
        print(f"\n\n====== EVALUATING ON {num_samples} TEST DATASET SAMPLES ======")
        
        # Select a subset of examples for evaluation
        if len(test_dataset) > num_samples:
            indices = torch.randperm(len(test_dataset))[:num_samples].tolist()
            eval_samples = [test_dataset[i] for i in indices]
        else:
            eval_samples = test_dataset[:num_samples]
        
        success_count = 0
        thinking_count = 0
        
        for i, sample in enumerate(eval_samples):
            print(f"Evaluating sample {i+1}/{len(eval_samples)}")
            
            input_text = sample["text"]
            
            model_start = input_text.find("<start_of_turn>model")
            if model_start == -1:
                print(f"Skipping sample {i+1}: couldn't find model start marker")
                continue
                
            prompt = input_text[:model_start + len("<start_of_turn>model")]
            
            try:
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(f"cuda:{cuda_device}") for k, v in inputs.items()}
                
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=500, 
                    temperature=0.01, 
                    top_p=0.95,
                    repetition_penalty=1.0,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
                
                has_tool_call = "<tool_call>" in generated_text
                has_thinking = "<think>" in generated_text
                
                if has_tool_call:
                    success_count += 1
                if has_thinking:
                    thinking_count += 1
                    
                # Print sample outputs for manual inspection
                if i < 3: 
                    print(f"\nSample {i+1} output:")
                    print(generated_text[:500] + "..." if len(generated_text) > 500 else generated_text)
                    print("-" * 50)
            
            except Exception as e:
                print(f"Error processing sample {i+1}: {str(e)}")
        
        if len(eval_samples) > 0:
            print(f"\nResults on {len(eval_samples)} test samples:")
            print(f"Tool call success rate: {success_count/len(eval_samples):.2%}")
            print(f"Thinking inclusion rate: {thinking_count/len(eval_samples):.2%}")
        else:
            print("No valid samples were evaluated.")

    # Run evaluation on test dataset
    evaluate_on_test_dataset(model, dataset["test"])

    print("\n\n====== FINAL EVALUATION SUMMARY ======")
    print("1. Finetuned vs Base Model:")
    for metric in ["function_accuracy", "params_accuracy", "values_accuracy", "thinking_rate"]:
        improvement = ft_metrics[metric] - base_metrics[metric]
        print(f"  - {metric.replace('_', ' ').title()}: {improvement:.2%} improvement")

    print("\n2. Key success indicators:")
    if ft_metrics["thinking_rate"] > 0.7:
        print("Thinking process inclusion is working well")
    elif ft_metrics["thinking_rate"] > 0.3:
        print("Thinking process is present but inconsistent")
    else:
        print("Thinking process is rarely included - finetuning may have failed")

except Exception as e:
    print(f"Error occurred during evaluation: {str(e)}")
    import traceback
    traceback.print_exc()