import os
from tqdm import tqdm
from datetime import datetime
import torch
from transformers import TrainingArguments, Trainer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

from utils import build_dataset, print_number_of_trainable_model_parameters


TRAINING_DATA_PATH = os.path.join(os.path.dirname(__file__), '..\\training_data')


def full_fine_tune_model(dataset_name, model_name, tokenize_function):
    # prepare the dataset
    tokenized_datasets = build_dataset(
        model_name=model_name,
        dataset_name=dataset_name,
        input_min_text_length=0,
        input_max_text_length=1024,
        tokenize_function=tokenize_function,
        sub_sample=100,
        dataset_type=['train', 'validation'],
        remove_columns=['id', 'topic', 'dialogue', 'summary'])
    
    # Now utilize the built-in Hugging Face Trainer class.
    # Pass the preprocessed dataset with reference to the original model.
    # Other training parameters are found experimentally and there is no need to go into details about those at the moment.
    training_args = TrainingArguments(
        output_dir=os.path.join(
            TRAINING_DATA_PATH,
            f"full_training-{datetime.strftime(datetime.now(), '%H%M%S')}"),
        learning_rate=1e-5,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=1,
        # save_steps=1,
        per_device_train_batch_size=4,
        max_steps=1)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cuda')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'])
    trainer.train()
    return trainer.model


def peft_fine_tune_model(dataset_name, model_name, tokenizer, tokenize_function):
    tokenized_datasets = build_dataset(
        model_name=model_name,
        dataset_name=dataset_name,
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
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cuda')
    peft_model = get_peft_model(model, lora_config)
    print(print_number_of_trainable_model_parameters(peft_model))

    peft_training_args = TrainingArguments(
        output_dir=os.path.join(
            TRAINING_DATA_PATH,
            f"peft_training-{datetime.strftime(datetime.now(), '%H%M%S')}"),
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

    peft_model_path=os.path.join(
        TRAINING_DATA_PATH,
        f"peft_training_checkpoint-{datetime.strftime(datetime.now(), '%H%M%S')}")
    peft_trainer.model.save_pretrained(peft_model_path)
    tokenizer.save_pretrained(peft_model_path)
    return peft_trainer.model


# TODO REMOVE
# def simple_train_model(
#     model, learning_rate, epochs, train_dataloader, valid_dataloader, evaluate_fn
# ):
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.01)

#     cum_loss_list=[]
#     acc_epoch=[]
#     acc_old=0
#     for epoch in tqdm(range(1, epochs + 1)):
#         model.train()
#         cum_loss=0
#         for idx, data_ in enumerate(train_dataloader):
#             optimizer.zero_grad()
#             label, input_ids = data_['labels'], data_['input_ids']
#             predicted_label = model(input_ids.to('cuda'))
#             loss = criterion(predicted_label, label.to('cuda'))
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
#             optimizer.step()
#             cum_loss+=loss.item()
#         cum_loss_list.append(cum_loss)
#         accu_val = evaluate_fn(model, valid_dataloader) if evaluate_fn is not None else 0
#         acc_epoch.append(accu_val)
#         # scheduler.step()

#         print(f"Epoch {epoch}, Loss: {cum_loss:.4f}, Validation Accuracy: {accu_val:.4f}")
#         if accu_val > acc_old:
#             acc_old = accu_val
#             model_path=os.path.join(
#                 TRAINING_DATA_PATH,
#                 f"simple_train_model-{datetime.strftime(datetime.now(), '%H%M%S')}.pth")
#             torch.save(model.state_dict(), model_path)


def train_model(
        model, train_dataloader, optimizer=None, criterion=None, 
        scheduler=None, clip=0.1, learning_rate=0.1, is_seq2seq=False,
        valid_dataloader=None, evaluate_fn=None, epochs=1):
    """
    Unified training function that handles both general and seq2seq model training.
    
    Args:
        model: PyTorch model to train
        train_dataloader: DataLoader for training data
        optimizer: Optimizer for training
        criterion: Loss function
        clip: Gradient clipping value (default: 0.1)
        is_seq2seq: Whether the model is seq2seq (default: False)
        valid_dataloader: Optional validation dataloader
        evaluate_fn: Optional evaluation function
        epochs: Number of epochs to train (default: 1)
    """
    criterion = torch.nn.CrossEntropyLoss() if criterion is None else criterion
    optimizer = (torch.optim.SGD if optimizer is None else optimizer)(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.01) if scheduler is None else scheduler

    best_val_acc = 0
    for epoch in tqdm(range(1, epochs + 1), desc="Training"):
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=False):
            optimizer.zero_grad()
            
            label, input_ids = batch['labels'].to('cuda'), batch['input_ids'].to('cuda')
            if is_seq2seq:
                output = model(input_ids, label)
                output = output[1:].flatten(0, 1)
                label = label[1:].flatten(0, 1)
            else:
                output = model(input_ids)
            
            loss = criterion(output, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
        
        # scheduler.step()
        avg_loss = epoch_loss / len(train_dataloader)
        
        # Validation if provided
        if valid_dataloader and evaluate_fn:
            val_acc = evaluate_fn(model, valid_dataloader, criterion, is_seq2seq)
            eval_loss = val_acc['loss']
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Validation Loss: {eval_loss:.4f}")
            
            # Save best model
            if eval_loss < best_val_acc:
                best_val_acc = eval_loss
                model_path = os.path.join(
                    TRAINING_DATA_PATH,
                    f"model_checkpoint-{datetime.strftime(datetime.now(), '%H%M%S')}.pth")
                torch.save(model.state_dict(), model_path)
        else:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    return model
