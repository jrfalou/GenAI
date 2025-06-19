import math
import random
import time
import torch
import torch.nn as nn

from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from utils import set_seed, print_number_of_trainable_model_parameters
from utils.eval_model import evaluate_model
from utils.training import train_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(self, vocab_len, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_len, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_batch):
        # input_batch = [src len, batch size]
        embed = self.dropout(self.embedding(input_batch))
        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        outputs, (hidden, cell) = self.lstm(embed)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor, hidden, cell):
        # input_tensor = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input_tensor = input_tensor.unsqueeze(0)  # [1, batch size]

        embedded = self.dropout(self.embedding(input_tensor))  # [1, batch size, emb dim]

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        prediction_logit = self.fc_out(output.squeeze(0))
        prediction = self.softmax(prediction_logit)  # [batch size, output dim]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
            # if 'weight' in name:
            #     nn.init.xavier_uniform_(param)
            # else:
            #     nn.init.zeros_(param)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 you use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        hidden = hidden.to(device)
        cell = cell.to(device)

        #first input to the decoder is the <bos> tokens
        input = trg[0,:]
        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if you are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from your predictions
            top1 = output.argmax(1)

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            #input = trg[t] if teacher_force else top1
            input = trg[t] if teacher_force else top1
        return outputs


def generate_translation(model, src_sentence, tokenizer, max_len=50):
    model.eval()
    with torch.no_grad():
        src_tensor = tokenizer.encode(src_sentence, add_special_tokens=True)
        src_tensor = torch.tensor(src_tensor).unsqueeze(1).to(device)

        # Pass the source tensor through the encoder
        hidden, cell = model.encoder(src_tensor)

        # Create a tensor to store the generated translation
        # get_stoi() maps tokens to indices
        trg_indexes = [tokenizer.bos_token_id]  # Start with <bos> token

        # Convert the initial token to a PyTorch tensor
          # Add batch dimension
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(1).to(model.device)

        # Generate the translation
        for _ in range(max_len):
            # Pass the target tensor and the previous hidden and cell states through the decoder
            output, hidden, cell = model.decoder(trg_tensor[-1], hidden, cell)
            pred_token = output.argmax(1)[-1].item()
            trg_indexes.append(pred_token)

            # If the predicted token is the <eos> token, stop generating
            if pred_token == tokenizer.eos_token_id:
                break

            # Convert the predicted token to a PyTorch tensor
            # Add batch dimension
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(1).to(model.device)

        # Convert the generated tokens to text
        # get_itos() maps indices to tokens
        trg_tokens = [tokenizer.decode(i) for i in trg_indexes]

        # Remove the <sos> and <eos> from the translation
        if trg_tokens[0] == '<bos>':
            trg_tokens = trg_tokens[1:]
        if trg_tokens[-1] == '<eos>':
            trg_tokens = trg_tokens[:-1]

        # Return the translation list as a string
        translation = " ".join(trg_tokens)
        return translation


class Seq2SeqCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id

    def __call__(self, batch):
        src_batch, tgt_batch = [], []

        for example in batch:
            src_text = example["de"].strip()
            tgt_text = example["en"].strip()

            # Add special tokens manually if needed
            src_ids = self.tokenizer.encode(src_text, add_special_tokens=True)
            tgt_ids = self.tokenizer.encode(tgt_text, add_special_tokens=True)

            src_batch.append(torch.tensor(src_ids))
            tgt_batch.append(torch.tensor(tgt_ids))

        # Pad and batch
        src_padded = pad_sequence(src_batch, padding_value=self.pad_id)
        tgt_padded = pad_sequence(tgt_batch, padding_value=self.pad_id)

        return {
            'input_ids': src_padded.to(device),
            'labels': tgt_padded.to(device)}


if __name__ == '__main__':
    # load the dataset
    dataset = load_dataset("bentrevett/multi30k")

    # set seed
    set_seed()

    # create tokenizers
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")

    # create dataloaders
    train_dataloader = DataLoader(
        dataset['train'], batch_size=4, shuffle=True,
        collate_fn=Seq2SeqCollator(tokenizer))
    val_dataloader = DataLoader(
        dataset['validation'], batch_size=4, shuffle=True,
        collate_fn=Seq2SeqCollator(tokenizer))

    # sample data
    data_ = next(iter(train_dataloader))
    print(data_)
    print(data_['input_ids'].shape, data_['labels'].shape)

    # create model
    model = Seq2Seq(
        encoder=Encoder(
            vocab_len=tokenizer.vocab_size,
            emb_dim=128,
            hid_dim=256,
            n_layers=1,
            dropout=0.3).to(device),
        decoder=Decoder(
            output_dim=tokenizer.vocab_size,
            emb_dim=128,
            hid_dim=256,
            n_layers=1,
            dropout=0.3).to(device))
    print(
        f'Seq2Seq model parameters to be updated:\n'
        f'{print_number_of_trainable_model_parameters(model)}\n')
    
    # Actual translation: Asian man sweeping the walkway.
    src_sentence = 'Ein asiatischer Mann kehrt den Gehweg.'
    generated_translation = generate_translation(
        model, src_sentence=src_sentence, tokenizer=tokenizer, max_len=12)
    print(generated_translation)

    # train model
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        is_seq2seq=True,
        epochs=5, clip=1, learning_rate=0.001,
        optimizer=torch.optim.Adam,
        criterion=criterion,
        valid_dataloader=val_dataloader,
        evaluate_fn=evaluate_model)   

    # Actual translation: Asian man sweeping the walkway.
    generated_translation = generate_translation(
        model, src_sentence=src_sentence, tokenizer=tokenizer, max_len=12)
    print(generated_translation)
