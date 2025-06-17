import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

from utils.training import train_model


def predict(text, tokenizer, model, ag_news_label):
    with torch.no_grad():
        # Convert text to tensor and move to GPU
        text = torch.tensor(tokenizer(text)['input_ids']).unsqueeze(dim=0).to('cuda')
        output = model(text)
        return ag_news_label[output.argmax(1).item()]


def evaluate(model, dataloader):
    model.eval()
    total_acc, total_count= 0, 0
    with torch.no_grad():
        for idx, data_ in enumerate(dataloader):
            label, input_ids = data_['labels'], data_['input_ids']
            predicted_label = model(input_ids.to('cuda'))
            total_acc += (predicted_label.argmax(1) == label.to('cuda')).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        # self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    # input_ids is expected to be of shape (batch_size, sequence_length)
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
        return self.fc(embedded)


if __name__ == "__main__":
    dataset = load_dataset("ag_news")
    dataset_splits = dataset['train'].train_test_split(
        test_size=0.1, shuffle=True, seed=42)
    dataset['train'] = dataset_splits['train']
    dataset['validation'] = dataset_splits['test']

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format(type="torch")

    BATCH_SIZE = 64
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    train_dataloader = DataLoader(
        tokenized_dataset['train'], batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
    valid_dataloader = DataLoader(
        tokenized_dataset['validation'], batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
    test_dataloader = DataLoader(
        tokenized_dataset['test'], batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)

    # Create a simple text classification model
    emsize = BATCH_SIZE
    vocab_size = len(tokenizer.get_vocab())
    num_class = len(dataset["train"].features["label"].names)
    model = TextClassificationModel(vocab_size, emsize, num_class).to('cuda')

    texts = next(iter(valid_dataloader))['input_ids']
    predicted_label = model(texts)
    print(f"Predicted label shape: {predicted_label.shape}")

    ag_news_label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    text = "The stock market crashed today, causing widespread panic."
    predicted_label = predict(text, tokenizer, model, ag_news_label)
    print(f"Predicted label for '{text}': {predicted_label}")

    # Evaluate the model on the validation set
    accuracy = evaluate(model, valid_dataloader)
    print(f"Validation accuracy: {accuracy:.4f}")

    # Evaluate the model on the test set
    test_accuracy = evaluate(model, test_dataloader)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Set up training parameters
    # train_model(
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        epochs=10,
        learning_rate=0.1,
        evaluate_fn=evaluate)

    # Evaluate the model on the test set
    test_accuracy = evaluate(model, test_dataloader)
    print(f"Test accuracy: {test_accuracy:.4f}")
