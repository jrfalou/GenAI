import time
from utils import build_dataset, print_number_of_trainable_model_parameters
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType


def full_fine_tune_model(dataset, model, tokenize_function):
    # prepare the dataset
    tokenized_datasets = build_dataset(
        model_name=model.name,
        dataset_name=dataset.name,
        input_min_text_length=0,
        input_max_text_length=1024,
        tokenize_function=tokenize_function,
        sub_sample=100,
        remove_columns=['id', 'topic', 'dialogue', 'summary'])
    
    # Now utilize the built-in Hugging Face Trainer class.
    # Pass the preprocessed dataset with reference to the original model.
    # Other training parameters are found experimentally and there is no need to go into details about those at the moment.
    output_dir = f'./training_data/dialogue-summary-training-{str(int(time.time()))}'
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-5,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=1,
        # save_steps=1,
        per_device_train_batch_size=4,
        max_steps=1)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'])
    trainer.train()
    return trainer


def peft_fine_tune_model(model, dataset, tokenizer, tokenize_function):
    tokenized_datasets = build_dataset(
        model_name=model.name,
        dataset_name=dataset.name,
        input_min_text_length=0,
        input_max_text_length=1024,
        tokenize_function=tokenize_function,
        sub_sample=100,
        remove_columns=['id', 'topic', 'dialogue', 'summary'])

    #  Setup the PEFT/LoRA model for Fine-Tuning
    lora_config = LoraConfig(
        r=32,  # Rank
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM)  # FLAN-T5
    
    # Add LoRA adapter layers/parameters to the original LLM to be trained
    peft_model = get_peft_model(model, lora_config)
    print(print_number_of_trainable_model_parameters(peft_model))

    output_dir = f'./training_data/peft-dialogue-summary-training-{str(int(time.time()))}'
    peft_training_args = TrainingArguments(
        output_dir=output_dir,
        auto_find_batch_size=True,
        learning_rate=1e-3, # Higher learning rate than full fine-tuning.
        # num_train_epochs=1,
        logging_steps=1,
        max_steps=1000)
        
    peft_trainer = Trainer(
        model=peft_model,
        args=peft_training_args,
        train_dataset=tokenized_datasets["train"])
    peft_trainer.train()

    peft_model_path="./training_data/peft-dialogue-summary-checkpoint-local"
    peft_trainer.model.save_pretrained(peft_model_path)
    tokenizer.save_pretrained(peft_model_path)
    return peft_trainer
