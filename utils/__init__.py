import re
import random
import torch
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
from nltk.tokenize import word_tokenize
from collections import Counter


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return (
        f"trainable model parameters: {trainable_model_params}\n"
        f"all model parameters: {all_model_params}\npercentage of "
        f"trainable model parameters: "
        f"{100 * trainable_model_params / all_model_params:.2f}%")


def build_dataset(model_name,
                  dataset_name,
                  tokenize_function,
                  dataset_type=None,
                  sub_sample=None,
                  remove_columns=None,
                  filter_fn=None):

    """
    Preprocess the dataset and split it into train and test parts.

    Parameters:
    - model_name (str): Tokenizer model name.
    - dataset_name (str): Name of the dataset to load.
        
    Returns:
    - dataset_splits (datasets.dataset_dict.DatasetDict): Preprocessed dataset containing train and test parts.
    """
    
    dataset = load_dataset(dataset_name, split=dataset_type)
    if dataset_type is not None:
        if not isinstance(dataset_type, list):
            dataset_type = [dataset_type]
            dataset = [dataset]
        dataset = DatasetDict({s: d for (s, d) in zip(dataset_type, dataset)})
    
    # Filter the dialogues of length between input_min_text_length and input_max_text_length characters.
    if filter_fn is not None:
        dataset = dataset.filter(lambda x: filter_fn(x), batched=True)

    # Prepare tokenizer. Setting device_map="auto" allows to switch between GPU and CPU automatically.
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    
    # Tokenize each dialogue.
    tokenized_datasets = dataset.map(
        tokenize_function, fn_kwargs={'tokenizer': tokenizer}, batched=True)
    tokenized_datasets.set_format(type="torch")

    if remove_columns:
        # remove columns that are not needed for training
        tokenized_datasets = tokenized_datasets.remove_columns(remove_columns)

    if sub_sample:
        # sub-sample the dataset (only keep 1% of the data if sub_sample==100)
        tokenized_datasets = tokenized_datasets.filter(
            lambda example, index: [i % sub_sample == 0 for i in index],
            with_indices=True, batched=True)

    return tokenized_datasets


def compare_model_weights(model1, model2, rtol=1e-5, atol=1e-8):
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2:
            print(f"Parameter name mismatch: {name1} != {name2}")
            return False
        if not torch.allclose(param1.data, param2.data, rtol=rtol, atol=atol):
            print(f"Parameter '{name1}' differs.")
            return False
    return True


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)
    return s


def preprocess_words(words):
    tokens = word_tokenize(words)
    tokens = [preprocess_string(w) for w in tokens]
    return [w.lower() for w in tokens if len(w) != 0 or not(w in string.punctuation)]


def build_vocab(dataset, tokenizer):
    # Build vocabulary from the dataset [{"text": "example text"}]
    # This is a simple example, in practice you might want to use a more sophisticated tokenizer
    counter = Counter()
    for example in dataset:
        tokens = tokenizer.tokenize(example["text"].lower())
        counter.update(tokens)

    vocab = {"<unk>": 0}
    for idx, (token, _) in enumerate(counter.most_common(), start=1):
        vocab[token] = idx

    # Reverse lookup
    itos = {idx: token for token, idx in vocab.items()}
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sample tokens: {list(vocab.keys())[:10]}")
    return lambda x: [vocab[u] for u in x], itos
