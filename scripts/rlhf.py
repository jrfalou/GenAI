
import torch
from transformers import AutoModelForSeq2SeqLM
from trl import AutoModelForSeq2SeqLMWithValueHead
from peft import PeftModel, LoraConfig, TaskType

from libs.utils import build_dataset, print_number_of_trainable_model_parameters


def tokenize_function(sample, tokenizer):
    # Wrap each dialogue with the instruction.
    prompt = f"""
Summarize the following conversation.

{sample["dialogue"]}

Summary:
"""
    sample["input_ids"] = tokenizer.encode(prompt)
    
    # This must be called "query", which is a requirement of our PPO library.
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample


if __name__ == '__main__':
    model_name="google/flan-t5-base"
    huggingface_dataset_name = "knkarthick/dialogsum"

    # load dataset (only "train" part will be enough for this lab).    
    dataset = build_dataset(
        model_name=model_name,
        dataset_name=huggingface_dataset_name,
        input_min_text_length=200, 
        input_max_text_length=1000,
        tokenize_function=tokenize_function,
        dataset_type="train")
    # Split the dataset into train and test parts.
    dataset_splits = dataset.train_test_split(test_size=0.2, shuffle=False, seed=42)

    lora_config = LoraConfig(
        r=32, # Rank
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16)

    peft_model = PeftModel.from_pretrained(
        model, 
        './Lab3/peft-dialogue-summary-checkpoint-from-s3/', 
        lora_config=lora_config,
        torch_dtype=torch.bfloat16, 
        device_map="auto",                                       
        is_trainable=True)

    print(
        f'PEFT model parameters to be updated:\n'
        '{print_number_of_trainable_model_parameters(peft_model)}\n')

    # Fine-tune the PEFT model.
    ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(peft_model,                                                               
                                                               torch_dtype=torch.bfloat16,
                                                               is_trainable=True)

    print(f'PPO model parameters to be updated (ValueHead + 769 params):\n{print_number_of_trainable_model_parameters(ppo_model)}\n')
    print(ppo_model.v_head)