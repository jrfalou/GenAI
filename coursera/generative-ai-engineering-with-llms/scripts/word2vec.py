import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils.training import train_model


# Set the device to GPU if available; otherwise, use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CbowCollator:
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
        for i in range(self.context_size, batch_size - self.context_size):
            target.append(batch[i])
            context.append([batch[i - j] for j in range(self.context_size, 0, -1)]
                           + [batch[i + j + 1] for j in range(0, self.context_size)])
        return {
            'input_ids': torch.tensor(context, dtype=torch.int64).to(DEVICE),
            'labels': torch.tensor(target).to(DEVICE).reshape(-1)}


class CBOW(nn.Module):
    # Initialize the CBOW model
    def __init__(self, vocab_size, embed_dim):
        
        super(CBOW, self).__init__()
        # Define the embedding layer using nn.EmbeddingBag or nn.Embedding
        # It outputs the average of context words embeddings
        # self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Define the first linear layer with input size embed_dim and output size embed_dim//2
        self.linear1 = nn.Linear(embed_dim, embed_dim//2)
        # Define the fully connected layer with input size embed_dim//2 and output size vocab_size
        self.fc = nn.Linear(embed_dim//2, vocab_size)
        self.init_weights()

    # Initialize the weights of the model's parameters
    def init_weights(self):
        # Initialize the weights of the embedding layer
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # Initialize the weights of the fully connected layer
        self.fc.weight.data.uniform_(-initrange, initrange)
        # Initialize the biases of the fully connected layer to zeros
        self.fc.bias.data.zero_()

    def forward(self, input_ids):
        if isinstance(self.embedding, nn.EmbeddingBag):
            # If using EmbeddingBag, we need to provide offsets
            text = input_ids.flatten().to('cuda')  # Flatten the input_ids and move to GPU
            offsets = torch.tensor(
                [i for i in range(0, len(text), int(len(text) / input_ids.shape[0]))],
                device='cuda')
            embedded = self.embedding(text, offsets)
        else:
            # If using standard Embedding, we can directly pass input_ids
            embedded = self.embedding(input_ids.to('cuda')).mean(dim=1)
        # Apply the ReLU activation function to the output of the first linear layer
        out = torch.relu(self.linear1(embedded))
        # Pass the output of the ReLU activation through the fully connected layer
        return self.fc(out)


class SkipGramCollator:
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
        for i in range(self.context_size, batch_size - self.context_size):
            target.append([batch[i]] * (2 * self.context_size))
            for j in range(self.context_size):
                context.append(batch[i - j - 1])
                context.append(batch[i + j + 1])
        return {
            'input_ids': torch.tensor(context, dtype=torch.int64).to(DEVICE),
            'labels': torch.tensor(target).to(DEVICE).reshape(-1)}


class SkipGram_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGram_Model, self).__init__()
        # Define the embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim)
        
        # Define the fully connected layer
        self.fc = nn.Linear(in_features=embed_dim, out_features=vocab_size)

    def forward(self, text):
        # Perform the forward pass
        # Pass the input text through the embedding layer
        out = self.embedding(text)
        
        # Pass the output of the embedding layer through the fully connected layer
        # Apply the ReLU activation function
        out = torch.relu(out)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), '..\\Labs\\Lab4\\toy.txt'), 'r') as file:
        toy_data = file.read()

    # Ensure same length across batches for the cbow
    batch_size = 64  # Number of samples in a batch
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenized_toy = tokenizer(toy_data, add_special_tokens=False).input_ids
    padding = batch_size - len(tokenized_toy) % batch_size
    padded_toy = tokenizer.decode(
        tokenized_toy + tokenized_toy[:padding], skip_special_tokens=True)
    tokens = tokenizer(padded_toy, add_special_tokens=False)

    # Create a dataloader for the cboword model
    # It is context size dependent, so we need to ensure that the batch size is a multiple of context_size
    context_size = 3  # Number of previous words to consider as context
    data_collator = CbowCollator(tokenizer, context_size)
    dataloader = DataLoader(
        tokens.input_ids, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    # Create the model
    vocab_size = len(tokenizer.get_vocab())
    model = CBOW(
        vocab_size=vocab_size,
        embed_dim=24).to(DEVICE)
    
    # test the dataloader and the model
    data_ = next(iter(dataloader))
    print(data_['input_ids'], data_['labels'])
    out = model(data_['input_ids'])
    print("Output shape:", out.shape)  # Should be (batch_size - 2*context_size, vocab_size)

    # Train the model
    train_model(
        model=model,
        learning_rate=0.001,
        epochs=1000,
        train_dataloader=dataloader,
        valid_dataloader=dataloader,
        evaluate_fn=None)
    
    # Get the word embeddings from the model
    word_embeddings = model.embedding.weight.detach().cpu().numpy()
    print("Word embeddings shape:", word_embeddings.shape)  # Should be (vocab_size, embed_dim)
    print("Word embeddings:", word_embeddings)  # Print the word embeddings

    ## SkipGram model

    # Create a dataloader for the skipgram model
    # It is context size dependent, so we need to ensure that the batch size is a multiple of context_size
    context_size = 3  # Number of previous words to consider as context
    data_collator = SkipGramCollator(tokenizer, context_size)
    dataloader = DataLoader(
        tokens.input_ids, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    # Create the model
    vocab_size = len(tokenizer.get_vocab())
    model = SkipGram_Model(
        vocab_size=vocab_size,
        embed_dim=24).to(DEVICE)
    
    # test the dataloader and the model
    data_ = next(iter(dataloader))
    print(data_['input_ids'], data_['labels'])
    out = model(data_['input_ids'])
    print("Output shape:", out.shape)  # Should be (batch_size - 2*context_size, vocab_size)

    # Train the model
    train_model(
        model=model,
        learning_rate=0.001,
        epochs=1000,
        train_dataloader=dataloader,
        valid_dataloader=dataloader,
        evaluate_fn=None)
    
    # Get the word embeddings from the model
    word_embeddings = model.embedding.weight.detach().cpu().numpy()
    print("Word embeddings shape:", word_embeddings.shape)  # Should be (vocab_size, embed_dim)
    print("Word embeddings:", word_embeddings)  # Print the word embeddings
