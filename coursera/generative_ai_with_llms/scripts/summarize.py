import time
import random
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer)
from peft import PeftModel, PeftConfig

from utils import print_number_of_trainable_model_parameters
from utils.eval_model import get_rouge_results
from utils.training import full_fine_tune_model, peft_fine_tune_model


DASH_LINE = '-'.join('' for x in range(100))


def print_sample(ds, indices):
    for i, index in enumerate(indices):
        print(DASH_LINE)
        print('Example ', i + 1)
        print(DASH_LINE)
        print('INPUT DIALOGUE:')
        print(ds['test'][index]['dialogue'])
        print(DASH_LINE)
        print('BASELINE HUMAN SUMMARY:')
        print(ds['test'][index]['summary'])
        print(DASH_LINE)
        print()


def print_summary_result(summary_type, output, baseline):
    print(DASH_LINE)
    print(f'MODEL GENERATION - {summary_type.upper()} SHOT:\n{output}')
    print(DASH_LINE)
    print(f"BASELINE HUMAN SUMMARY:\n{baseline}\n")


def make_shot_inference_prompt(dataset, nb_shots, to_summarize):
    example_indices = random.sample(range(1, dataset.num_rows), nb_shots)
    examples=[
        {
            'raw_txt': dataset[index_]['dialogue'],
            'result': dataset[index_]['summary']}
        for index_ in example_indices]

    prompt = ''
    for example in examples:
        dialogue, summary = example['raw_txt'], example['result']
        prompt += f"Dialogue:\n{dialogue}\nWhat was going on?\n{summary}\n\n\n"
        
    prompt += f"Dialogue:\n{to_summarize}\nWhat was going on?\n"
    return prompt


def model_summarize(prompt, model, tokenizer, max_new_tokens=50, do_sample=True, temperature=0.1, human_baseline_output=None):
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)

    # run the model
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    return tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            generation_config=generation_config,
        )[0], 
        skip_special_tokens=True)


def tokenize_function(example, tokenizer):
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    return example


if __name__ == '__main__':
    # get dataset
    huggingface_dataset_name = "knkarthick/dialogsum"
    dataset = load_dataset(huggingface_dataset_name)

    example_indices = [40, 200]
    print_sample(ds=dataset, indices=example_indices)

    # get model
    model_name='google/flan-t5-base'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cuda')
    print(print_number_of_trainable_model_parameters(model))

    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # try and summarize a dialogue
    to_summarize_index = 200
    to_summarize = dataset['test'][to_summarize_index]['dialogue']

    # try different kinds of configs/prompts
    # get zero shot prompt
    zero_shot_prompt = make_shot_inference_prompt(
        dataset=dataset['test'],
        nb_shots=0, to_summarize=to_summarize)
    output = model_summarize(
        prompt=zero_shot_prompt,
        model=model, tokenizer=tokenizer,
        max_new_tokens=50, do_sample=True, temperature=0.1)
    print_summary_result('zero', output, dataset['test'][to_summarize_index]['summary'])

    # get few shot prompt
    few_shot_prompt = make_shot_inference_prompt(
        dataset=dataset['test'],
        nb_shots=5, to_summarize=to_summarize)
    output = model_summarize(
        prompt=few_shot_prompt,
        model=model, tokenizer=tokenizer,
        max_new_tokens=50, do_sample=True, temperature=0.1)
    print_summary_result('few', output, dataset['test'][to_summarize_index]['summary'])

    # Evaluate the Model Quantitatively (with ROUGE Metric)
    original_eval = get_rouge_results(dataset, model, tokenizer)
    print(original_eval)

    #### fine-tune the model ####
    fine_tuned_model = full_fine_tune_model(
        huggingface_dataset_name, model_name, tokenize_function)
    full_fine_tune_eval = get_rouge_results(dataset, fine_tuned_model, tokenizer)
    print(full_fine_tune_eval)

    ### PEFT (Lora) Model Fine-Tuning ###
    peft_tuned_model = peft_fine_tune_model(huggingface_dataset_name, model_name, tokenizer, tokenize_function)
    peft_fine_tune_eval = get_rouge_results(dataset, peft_tuned_model, tokenizer)
    print(peft_fine_tune_eval)

    # Prepare this model by adding an adapter to the original FLAN-T5 model.
    # You are setting is_trainable=False because the plan is only to perform inference with this PEFT model.
    # If you were preparing the model for further training, you would set is_trainable=True.
    # peft_model = PeftModel.from_pretrained(
    #     model, 
    #     './training_data/peft-dialogue-summary-checkpoint-local/',
    #     torch_dtype=torch.bfloat16,
    #     is_trainable=False)
