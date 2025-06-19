import os
import torch
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from peft import PeftModel, LoraConfig, TaskType
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, GenerationConfig, DataCollatorWithPadding)
from trl import create_reference_model, PPOConfig, PPOTrainer
from trl.core import LengthSampler

from utils import build_dataset, print_number_of_trainable_model_parameters
from utils.eval_model import evaluate_toxicity

# from generative_ai_with_llms.debug_libs.ppo_trainer import PPOTrainer
from generative_ai_with_llms.libs.modeling_value_head import (
    MyAutoModelForSeq2SeqLMWithValueHead, MyT5WithOverrides, ModelWrapper)


TRAINING_DATA_PATH = './training_data'
os.environ['DO_PRINT_MEMORY'] = 'False'


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
    def filter_fn(x):
        # Filter the dialogues of length between input_min_text_length and input_max_text_length characters.
        return [
            len(dialogue) > 200
            and len(dialogue) <= 1000 for dialogue in x['dialogue']]
    dataset = build_dataset(
        model_name=model_name,
        dataset_name=huggingface_dataset_name,
        tokenize_function=tokenize_function,
        dataset_type="train",
        filter_fn=filter_fn)

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
    model = MyT5WithOverrides.from_pretrained_custom(model_name, torch_dtype=torch.bfloat16)
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
        os.path.join(
            os.path.dirname(__file__), '..', 'Labs', 'Lab3',
            'peft-dialogue-summary-checkpoint-from-s3'), 
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
    ppo_model = MyAutoModelForSeq2SeqLMWithValueHead.from_pretrained(
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
    toxicity_model = ModelWrapper.from_pretrained_custom(toxicity_model_name, device_map="auto")
    print(toxicity_model.config.id2label)
    
    # Perform the calculation of the model toxicity before fine-tuning/detoxification
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    mean_before_detoxification, std_before_detoxification = evaluate_toxicity(
        model=ref_model, 
        toxicity_model_name=toxicity_model_name, 
        tokenizer=tokenizer, 
        dataset=dataset_splits["test"], 
        num_samples=100)
    print(f'toxicity [mean, std] before detox: '
          f'[{mean_before_detoxification}, {std_before_detoxification}]')

    ### Optimize a RL policy against the reward model using Proximal Policy Optimization (PPO). ###
    # PPO is a policy gradient method for reinforcement learning.
    # PPO configuration
    # Parameter	                    Source	                    Used In Code As	                Meaning & Role
    # -------------------------------------------------------------------------------------------------------------------------------
    # total_episodes	            PPOConfig (rlhf.py)	        args.total_episodes	            Total number of training queries (not updates or batches). 
    #                                                                                           If unset, it's computed as num_train_epochs * dataset_len.
    # num_train_epochs	            PPOConfig	                args.num_train_epochs	        Used only to compute total_episodes if not set explicitly.
    # per_device_train_batch_size	PPOConfig	                args.per_device_train_batch_size	Number of samples per device in a forward pass.
    # gradient_accumulation_steps	PPOConfig or default	    args.gradient_accumulation_steps	Controls global batch size and effective optimization step rate.
    # num_mini_batches	            PPOConfig or default (4)	args.num_mini_batches	        Each rollout batch is split into this many mini-batches.
    # num_ppo_epochs	            PPOConfig	                args.num_ppo_epochs	            Number of optimization passes over each rollout batch.
    # num_total_batches	            Computed internally	        args.num_total_batches	        Total PPO updates = ceil(total_episodes / batch_size).
    # batch_size	                Computed internally	        args.batch_size	                Global batch size = per_device_batch * grad_accum * world_size.
    # local_batch_size	            Computed internally	        args.local_batch_size	        Local batch size per process.
    # mini_batch_size	            Computed internally	        args.mini_batch_size	        Global mini-batch size = batch_size / num_mini_batches.
    # local_mini_batch_size	        Computed internally	        args.local_mini_batch_size	    Local version of above for distributed training.
    # num_sample_generations	    PPOConfig	                args.num_sample_generations	    If > 0, triggers evaluation generations at regular intervals.
    # response_length	            PPOConfig	                args.response_length	            Max number of tokens generated per completion.
    # stop_token_id	                PPOConfig	                args.stop_token_id	            Used to truncate generated responses during postprocessing.

    ppo_config = PPOConfig(
        output_dir=os.path.join(
            TRAINING_DATA_PATH,
            f"ppo_training-{datetime.strftime(datetime.now(), '%H%M%S')}"),
        learning_rate=1.41e-5,
        per_device_train_batch_size=4,
        # gradient_accumulation_steps=4,
        # num_mini_batches=4,
        num_train_epochs=4,
        # total_episodes=40,  # max_ppo_steps (10) * batch_size (4)
        response_length=200,  # max length for generation
        remove_unused_columns=False,
        logging_steps=10,
        stop_token_id=tokenizer.pad_token_id,  # often 0 for T5
        num_sample_generations=0,  # set to 1 to generate evaluation samples during training. Also need an evaluation dataset.
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3)

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
    
    # Perform the calculation of the model toxicity after fine-tuning/detoxification
    mean_after_detoxification, std_after_detoxification = evaluate_toxicity(
        model=ppo_model,
        toxicity_model_name=toxicity_model_name, 
        tokenizer=tokenizer, 
        dataset=dataset_splits["test"], 
        num_samples=100)
    print(f'toxicity [mean, std] after detox: '
          f'[{mean_after_detoxification}, {std_after_detoxification}]')
    mean_improvement = (mean_before_detoxification - mean_after_detoxification) / mean_before_detoxification
    std_improvement = (std_before_detoxification - std_after_detoxification) / std_before_detoxification
    print(f'Percentage improvement of toxicity score after detoxification:')
    print(f'mean: {mean_improvement*100:.2f}%')
    print(f'std: {std_improvement*100:.2f}%')

    # Evaluate the Model Qualitatively
    batch_size = 20
    compare_results = {}
    df_batch = dataset_splits["test"][0:batch_size]
    compare_results["query"] = df_batch["query"]
    prompt_tensors = df_batch["input_ids"]

    summary_tensors_ref = []
    summary_tensors = []
    gen_len = LengthSampler(100, 400)  # output_min_length, output_max_length
    # Get response from ppo and base model.
    for i in tqdm(range(batch_size)):
        gen_len_ = gen_len()
        generation_kwargs = {
            "min_length": 5,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "max_new_tokens" : gen_len_}
        
        summary = ref_model.generate(
            input_ids=torch.as_tensor(prompt_tensors[i]).unsqueeze(dim=0).to('cuda'), 
            **generation_kwargs
        ).sequences.squeeze()[prompt_tensors.shape[1]:]
        summary_tensors_ref.append(summary)

        summary = ppo_model.generate(
            input_ids=torch.as_tensor(prompt_tensors[i]).unsqueeze(dim=0).to('cuda'), 
            **generation_kwargs
        ).sequences.squeeze()[prompt_tensors.shape[1]:]
        summary_tensors.append(summary)

    # Decode responses.
    compare_results["response_before"] = [tokenizer.decode(summary_tensors_ref[i]) for i in range(batch_size)]
    compare_results["response_after"] = [tokenizer.decode(summary_tensors[i]) for i in range(batch_size)]

    # Sentiment analysis of query/response pairs before/after.
    reward_kwargs = {
        "top_k": None, # Return all scores.
        "function_to_apply": "none", # You want the raw logits without softmax.
        "batch_size": 16}
    sentiment_pipe = pipeline("sentiment-analysis", 
                              model=toxicity_model_name, 
                              device='cuda',
                              framework="pt")
    not_hate_index = 0
    texts_before = [d + s for d, s in zip(compare_results["query"], compare_results["response_before"])]
    rewards_before = sentiment_pipe(texts_before, **reward_kwargs)
    compare_results["reward_before"] = [reward[not_hate_index]["score"] for reward in rewards_before]

    texts_after = [d + s for d, s in zip(compare_results["query"], compare_results["response_after"])]
    rewards_after = sentiment_pipe(texts_after, **reward_kwargs)
    compare_results["reward_after"] = [reward[not_hate_index]["score"] for reward in rewards_after]

    df_compare_results = pd.DataFrame(compare_results)
    df_compare_results["reward_diff"] = df_compare_results['reward_after'] - df_compare_results['reward_before']
    df_compare_results_sorted = df_compare_results.sort_values(by=['reward_diff'], ascending=False).reset_index(drop=True)
    df_compare_results_sorted.to_csv(
        os.path.join(
            TRAINING_DATA_PATH,
            f"ppo_training-{datetime.strftime(datetime.now(), '%H%M%S')}-compare_results.csv"),
        index=False)

    print('done')
