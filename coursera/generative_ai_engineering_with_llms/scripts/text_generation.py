import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from utils.training import train_model


# Set the device to GPU if available; otherwise, use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def genngrams(tokens, context_size):
    """
    Generates n-grams from a list of tokens, where each n-gram consists of a 
    context (previous words) and a target (next word).

    The function constructs a list of tuples where:
    - The first element is a list of `CONTEXT_SIZE` previous words.
    - The second element is the target word that follows the context.

    Parameters:
    tokens (list): A list of preprocessed word tokens.
    context_size (int): The number of previous words to consider as context.

    Returns:
    list: A list of tuples representing n-grams.
          Each tuple contains (context_words, target_word).
    """

    # Generate n-grams
    # Iterate through the tokens starting from index context_size to the end
    # For each token at position 'i', extract the previous context_size words as context
    ngrams = [
        (
            [tokens[i - j - 1] for j in range(context_size)],  # Context words (previous words)
            tokens[i]  # Target word (the word to predict)
        )
        for i in range(context_size, len(tokens))
    ]
    return ngrams


def write_song(model, my_song, context_size, tokenizer, number_of_words=100):
    """
    Generates text using a trained n-gram language model.

    Given an initial text (`my_song`), the function generates additional words by 
    predicting the next word iteratively based on the trained model.

    Parameters:
    model (nn.Module): The trained n-gram language model.
    my_song (str): The initial seed text to start generating words.
    number_of_words (int): The number of words to generate (default: 100).

    Returns:
    str: The generated song lyrics as a string.
    """
    for i in range(number_of_words):
        tokens = tokenizer(my_song, add_special_tokens=False).input_ids  # Tokenize the initial text
        with torch.no_grad():
            context = torch.tensor([
                tokens[i - j] for j in range(context_size, 0, -1)]).to(DEVICE)
            word_idx = torch.argmax(model(context))
            my_song += " " + tokenizer.decode(word_idx.detach().item())
    return my_song  # Return the generated lyrics


class NgramCollator:
    def __init__(self, tokenizer, context_size):
        self.tokenizer = tokenizer
        self.context_size = context_size

    def __call__(self, batch):
        """
        Processes a batch of text data into input (context) and output (target) tensors
        for training a language model.

        The function extracts:
        - `context`: A list of word indices representing the context words for each target word.
        - `target`: A list of word indices representing the target word to predict.

        Parameters:
        batch (list): A list of tokenized words (int).

        Returns:
        tuple: Two PyTorch tensors: (context_tensor, target_tensor)
            - context_tensor: Tensor of shape (batch_size - context_size, context_size),
                containing the word indices of context words.
            - target_tensor: Tensor of shape (batch_size - context_size,),
                containing the word indices of target words.
        """
        batch_size = len(batch)  # Get the size of the batch
        context, target = [], [] # Initialize lists for context and target words
        for i in range(self.context_size, batch_size):
            target.append(batch[i])
            context.append([batch[i - j] for j in range(self.context_size, 0, -1)])
        return {
            'input_ids': torch.tensor(context).to(DEVICE),
            'labels': torch.tensor(target).to(DEVICE).reshape(-1)}


class NGramLanguageModeler(nn.Module):
    """
    A neural network-based n-gram language model that predicts the next word 
    given a sequence of context words.

    This model consists of:
    - An embedding layer that converts word indices into dense vector representations.
    - A fully connected hidden layer with ReLU activation.
    - An output layer that predicts the probability distribution over the vocabulary.

    Parameters:
    vocab_size (int): The number of unique words in the vocabulary.
    embedding_dim (int): The size of the word embeddings (vector representation of words).
    context_size (int): The number of previous words used as context to predict the next word.
    """

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()

        # Store context size and embedding dimension
        self.context_size = context_size
        self.embedding_dim = embedding_dim

        # Embedding layer: Maps word indices to dense vectors
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Fully connected hidden layer: Maps the concatenated embeddings to a 128-dimensional space
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)

        # Output layer: Maps the hidden layer output to vocabulary size (probability distribution over words)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        """
        Forward pass of the model.

        Parameters:
        inputs (Tensor): A tensor of shape (batch_size, context_size) containing word indices.

        Returns:
        Tensor: A tensor of shape (batch_size, vocab_size) representing predicted probabilities for the next word.
        """

        # Convert input word indices into dense vectors using the embedding layer
        embeds = self.embeddings(inputs)  # Shape: (batch_size, context_size, embedding_dim)

        # Reshape the embeddings into a single vector per input sample
        embeds = torch.reshape(embeds, (-1, self.context_size * self.embedding_dim))  
        # New shape: (batch_size, context_size * embedding_dim)

        # Apply first fully connected layer with ReLU activation
        out = F.relu(self.linear1(embeds))  # Shape: (batch_size, 128)

        # Apply second fully connected layer to generate vocabulary-size logits
        out = self.linear2(out)  # Shape: (batch_size, vocab_size)

        return out


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), '..\\Labs\\Lab3\\song.txt'), 'r') as file:
        song = file.read()

    # Ensure same length across batches for the n-grams
    batch_size = 10  # Number of samples in a batch
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenized_song = tokenizer(song, add_special_tokens=False).input_ids
    padding = batch_size - len(tokenized_song) % batch_size
    padded_song = tokenizer.decode(
        tokenized_song + tokenized_song[:padding], skip_special_tokens=True)
    tokens = tokenizer(padded_song, add_special_tokens=False)
    
    # Create a dataloader for the n-grams
    # It is context size dependent, so we need to ensure that the batch size is a multiple of context_size
    context_size = 3  # Number of previous words to consider as context
    data_collator = NgramCollator(tokenizer, context_size)
    dataloader = DataLoader(
        tokens.input_ids, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    # Create the model
    vocab_size = len(tokenizer.get_vocab())
    model = NGramLanguageModeler(
        vocab_size=vocab_size,
        embedding_dim=batch_size,
        context_size=context_size).to(DEVICE)
    
    # test the dataloader and the model
    data_ = next(iter(dataloader))
    print(data_['input_ids'], data_['labels'])
    out = model(data_['input_ids'])
    print("Output shape:", out.shape)  # Should be (batch_size, vocab_size)

    # Generate a song using the model
    my_song = song.split('\n')[0]  # Start with the first line of the song
    my_song = write_song(
        model=model,
        my_song=my_song,
        context_size=context_size,
        tokenizer=tokenizer,
        number_of_words=100)
    print(my_song)  # Print the generated song lyrics

    # Train the model
    train_model(
        model=model,
        learning_rate=0.001,
        epochs=100000,
        train_dataloader=dataloader,
        valid_dataloader=dataloader,
        evaluate_fn=None)

    # Generate a song using the trained model
    my_song = song.split('\n')[0]  # Start with the first line of the song
    my_song = write_song(
        model=model,
        my_song=my_song,
        context_size=context_size,
        tokenizer=tokenizer,
        number_of_words=100)
    print(my_song)  # Print the generated song lyrics
