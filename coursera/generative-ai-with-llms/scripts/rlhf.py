import os
from datetime import datetime
import torch
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, GenerationConfig, DataCollatorWithPadding)
from trl import AutoModelForSeq2SeqLMWithValueHead, create_reference_model, PPOConfig  # , PPOTrainer
from peft import PeftModel, LoraConfig, TaskType

from debug_libs.ppo_trainer import PPOTrainer

from libs.utils import build_dataset, print_number_of_trainable_model_parameters
from libs.eval_model import evaluate_toxicity


TRAINING_DATA_PATH = os.path.join(os.path.dirname(__file__), '..\\training_data')


def tokenize_function(sample, tokenizer):
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in sample["dialogue"]]
    sample['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    sample['labels'] = tokenizer(sample["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    sample["query"] = tokenizer.batch_decode(sample['input_ids'], skip_special_tokens=True)
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
    dataset_splits = dataset['train'].train_test_split(
        test_size=0.2, shuffle=False, seed=42)

    lora_config = LoraConfig(
        r=32, # Rank
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
    )

    # model is a T5ForConditionalGeneration model (in this case flan-t5-base),
    # enhanced via LoRA adapters, and using a classic encoder-decoder (seq2seq) transformer architecture.
    # T5 is designed for sequence-to-sequence tasks like summarization, translation, and question answering. It has:
    # 1. An encoder that reads and processes the input text.
    # 2. A decoder that generates output tokens based on encoder context and previous output tokens.
    # 3. Shared token embeddings to tie input/output vocabularies.
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16)
    print(
        f'model parameters that can be updated:\n'
        f'{print_number_of_trainable_model_parameters(model)}\n')

    # peft_model is PeftModelForSeq2SeqLM
    # -> functionally the same as the previous LoRA-enhanced T5ForConditionalGeneration model,
    # but wrapped differently to enable PEFT-specific handling and modularity.
    # The LoRA logic is now fully managed by PEFT:
    # 1. built-in support for saving/loading adapters.
    # 2. can activate/deactivate adapters dynamically.
    # 3. can merge adapters into the base model for export/inference
    peft_model = PeftModel.from_pretrained(
        model, 
        'coursera\generative-ai-with-llms\Labs\Lab3\peft-dialogue-summary-checkpoint-from-s3/', 
        lora_config=lora_config,
        torch_dtype=torch.bfloat16, 
        device_map="auto",                                       
        is_trainable=True)
    print(
        f'PEFT model parameters to be updated:\n'
        f'{print_number_of_trainable_model_parameters(peft_model)}\n')

    # Prepare to fine-tune the LLM using Reinforcement Learning (RL).
    # Prepare the Proximal Policy Optimization (PPO) model passing the instruct-fine-tuned PEFT model to it.
    # PPO will be used to optimize the RL policy against the reward model.
    ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        peft_model, torch_dtype=torch.bfloat16, is_trainable=True, device_map="auto")

    # During PPO, only a few parameters will be updated.
    # Specifically, the parameters of the ValueHead.
    # The number of trainable parameters can be computed as (number of input units + 1) * number of output units
    # where the +1 is for the bias term.
    print(
        f'PPO model parameters to be updated (ValueHead + 769 params):\n'
        f'{print_number_of_trainable_model_parameters(ppo_model)}\n')
    print(ppo_model.v_head)

    # Load a frozen version of the model ref_model. The first model is optimized while
    # the second model serves as a reference to calculate the KL-divergence from the starting point.
    # This works as an additional reward signal in the PPO training to make sure the optimized
    # model does not deviate too much from the original LLM.
    ref_model = create_reference_model(ppo_model).to('cuda')
    print(
        f'Reference model parameters to be updated:\n'
        f'{print_number_of_trainable_model_parameters(ref_model)}\n')

    # Let's use Meta AI's RoBERTa-based hate speech model for the reward model.
    # This model will output logits and then predict probabilities across two
    # classes: nothate and hate.
    # Create the instance of the required model class for the RoBERTa model.
    toxicity_model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
    toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_name, device_map="auto")
    toxicity_model = AutoModelForSequenceClassification.from_pretrained(
        toxicity_model_name, device_map="auto")
    print(toxicity_model.config.id2label)
    
    # Perform the calculation of the model toxicity before fine-tuning/detoxification
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    mean_before_detoxification, std_before_detoxification = evaluate_toxicity(
        model=ref_model, 
        toxicity_model_name=toxicity_model_name, 
        tokenizer=tokenizer, 
        dataset=dataset_splits["test"], 
        num_samples=10)
    print(f'toxicity [mean, std] before detox: '
          f'[{mean_before_detoxification}, {std_before_detoxification}]')

    ### Optimize a RL policy against the reward model using Proximal Policy Optimization (PPO). ###
    # PPO is a policy gradient method for reinforcement learning.
    # PPO configuration
    ppo_config = PPOConfig(
        output_dir=os.path.join(
            TRAINING_DATA_PATH,
            f'ppo_training-{datetime.strftime(datetime.now(), '%H:%M:%S')}'),
        learning_rate=1.41e-5,
        per_device_train_batch_size=4,
        num_train_epochs=4,
        total_episodes=40,  # max_ppo_steps (10) * batch_size (4)
        response_length=400,  # max length for generation
        remove_unused_columns=False,
        logging_steps=10,
        stop_token_id=tokenizer.pad_token_id,  # often 0 for T5
        num_sample_generations=0,  # set to 1 to generate evaluation samples during training. Also need an evaluation dataset.
        save_strategy="no")

    # PPO Trainer
    # ensure the dataset is in the right format for PPO training
    dataset_splits["train_list"] = dataset_splits["train"].map(
        lambda x: {
            k: [x_.tolist() if isinstance(x_, torch.Tensor) else x_ for x_ in x[k]]
            for k in ['input_ids', 'labels']
        },
        batched=True,
        remove_columns=['id', 'dialogue', 'summary', 'topic', 'query'])
    dataset_splits["train_list"].set_format(type=None)

    # For the PPOTrainer initialization, we need a collator.
    #  What is DataCollatorWithPadding used for?
    # It is a batch preparation utility in Hugging Face’s Trainer and PPO loops. It ensures that:
        # Sequences in a batch are padded to the same length,
        # Special tokens (like <pad>) are inserted properly,
        # Tensors are returned in a format your model can train on (e.g., return_tensors='pt' for PyTorch).
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # in order to use the PPOTrainer, we need to pass a valid eos_token_id to the generation_config.
    ppo_model.generation_config = GenerationConfig(eos_token_id=tokenizer.eos_token_id)
    ppo_model.model = peft_model
    ppo_model.base_model_prefix = 'model'
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=ppo_model,
        ref_model=ref_model,
        value_model=ppo_model,
        reward_model=toxicity_model,
        train_dataset=dataset_splits['train_list'],
        data_collator=data_collator)
    
    # The fine-tuning loop consists of the following main steps:
        # Get the query responses from the policy LLM (PEFT model).
        # Get sentiments for query/responses from hate speech RoBERTa model.
        # Optimize policy with PPO using the (query, response, reward) triplet.
    # The workflow looks like this:
    # 1. Generate output using peft_model (policy)
    # 2. Compute reward R using reward model
    # 3. Estimate V(s) using value head
    # 4. Compute KL = log π_new - log π_ref
    # 5. Adjust reward: R' = R - β * KL
    # 6. Compute advantage A = R' - V(s)
    # 7. Compute loss:
        # policy_loss = - (ratio * A)
        # value_loss  = (V(s) - R')²
    # 8. Backprop both losses, update:
        # - policy → peft_model (LoRA adapters)
        # - critic  → value head
    ppo_trainer.train()
    print('done')
