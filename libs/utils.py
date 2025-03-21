from transformers import AutoTokenizer
from datasets import load_dataset


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return (
        f"trainable model parameters: {trainable_model_params}\n"
        "all model parameters: {all_model_params}\npercentage of "
        "trainable model parameters: {100 * trainable_model_params "
        "/ all_model_params:.2f}%")


def build_dataset(model_name,
                  dataset_name,
                  input_min_text_length, 
                  input_max_text_length,
                  tokenize_function,
                  dataset_type=None,
                  sub_sample=None,
                  remove_columns=None):

    """
    Preprocess the dataset and split it into train and test parts.

    Parameters:
    - model_name (str): Tokenizer model name.
    - dataset_name (str): Name of the dataset to load.
    - input_min_text_length (int): Minimum length of the dialogues.
    - input_max_text_length (int): Maximum length of the dialogues.
        
    Returns:
    - dataset_splits (datasets.dataset_dict.DatasetDict): Preprocessed dataset containing train and test parts.
    """
    
    dataset = load_dataset(dataset_name, split=dataset_type)
    
    # Filter the dialogues of length between input_min_text_length and input_max_text_length characters.
    dataset = dataset.filter(lambda x: len(x["dialogue"]) > input_min_text_length and len(x["dialogue"]) <= input_max_text_length, batched=False)

    # Prepare tokenizer. Setting device_map="auto" allows to switch between GPU and CPU automatically.
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    
    # Tokenize each dialogue.
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type="torch")

    if remove_columns:
        # remove columns that are not needed for training
        tokenized_datasets = tokenized_datasets.remove_columns(remove_columns)

    if sub_sample:
        # sub-sample the dataset (only keep 1% of the data if sub_sample==100)
        tokenized_datasets = tokenized_datasets.filter(
            lambda example, index: index % sub_sample == 0, with_indices=True)

    return tokenized_datasets
