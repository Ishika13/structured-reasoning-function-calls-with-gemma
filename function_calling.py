import os
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from enum import Enum
from torch.distributed.elastic.multiprocessing.errors import record


# Define the main function to be executed
@record
def main():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        print(f"Process group initialized: rank={rank}, world_size={world_size}")
    else:
        world_size = 1
        rank = 0

    print(f"[Rank {rank}] Initializing")
    model_name = "google/gemma-2-2b-it"
    dataset_name = "Jofthomas/hermes-function-calling-thinking-V1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    seed = 42
    set_seed(seed)

    tokenizer.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn><eos>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"

    def preprocess(sample):
        messages = sample["messages"]
        first_message = messages[0]

        if first_message["role"] == "system":
            system_message_content = first_message["content"]
            messages[1]["content"] = system_message_content + "Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>\n\n" + messages[1]["content"]
            messages.pop(0)

        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


    print(f"[Rank {rank}] Loading dataset...")
    dataset = load_dataset(dataset_name)
    dataset = dataset.rename_column("conversations", "messages")
    dataset = dataset.map(preprocess, remove_columns="messages")
    dataset = dataset["train"].train_test_split(0.1)
    print(dataset)
    if rank == 0:
        print(dataset["train"][8]["text"])

    class ChatmlSpecialTokens(str, Enum):
        tools = "<tools>"
        eotools = "</tools>"
        think = "<think>"
        eothink = "</think>"
        tool_call="<tool_call>"
        eotool_call="</tool_call>"
        tool_response="<tool_reponse>"
        eotool_response="</tool_reponse>"
        pad_token = "<pad>"
        eos_token = "<eos>"
        @classmethod
        def list(cls):
            return [c.value for c in cls]

    print(f"[Rank {rank}] Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            pad_token=ChatmlSpecialTokens.pad_token.value,
            additional_special_tokens=ChatmlSpecialTokens.list()
        )
    tokenizer.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn><eos>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"[Rank {rank}] Initializing model...")
    if local_rank != -1:
        device_map = {"": local_rank}
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map=device_map,
        attn_implementation='eager'
    )
    model.resize_token_embeddings(len(tokenizer))

    model = prepare_model_for_kbit_training(model)

    rank_dimension = 8
    lora_alpha = 16
    lora_dropout = 0.05

    print(f"[Rank {rank}] Setting up training...")
    peft_config = LoraConfig(r=rank_dimension,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout,
                            target_modules=["gate_proj","q_proj","lm_head","o_proj","k_proj","embed_tokens","down_proj","up_proj","v_proj"],
                            task_type=TaskType.CAUSAL_LM)

    username="ishikakulkarni"
    output_dir = "func_call_gemma-2-2B-V0" 
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 1
    gradient_accumulation_steps = 32
    logging_steps = 5
    learning_rate = 1e-4 

    max_grad_norm = 1.0
    num_train_epochs = 2
    warmup_ratio = 0.1
    lr_scheduler_type = "cosine"
    max_seq_length = 1024

    # Update training configuration for DDP
    training_arguments = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_seq_length=max_seq_length,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        weight_decay=0.1,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard",
        bf16=True,
        hub_private_repo=False,
        push_to_hub=False,
        num_train_epochs=num_train_epochs,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=True,
        
        # Set local_rank explicitly to use DDP
        local_rank=local_rank,
        
        # Configure distributed training
        ddp_find_unused_parameters=False,
        
        # Evaluation and logging settings
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        logging_first_step=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print(f"[Rank {rank}] Training...")
    trainer.train()

    print(f"[Rank {rank}] Saving model...")
    trainer.save_model()

    # Only push to hub from the main process
    if rank == 0:
        print("Pushing to hub...")
        trainer.push_to_hub(f"{username}/{output_dir}")

        print("Pushing tokenizer to hub...")
        tokenizer.eos_token = "<eos>"
        tokenizer.push_to_hub(f"{username}/{output_dir}", token=True)

    # Clean up the distributed environment
    if local_rank != -1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()