import argparse
from functools import cache
import random
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
import torch.backends.cudnn

import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import pandas as pd
import time
import os
from pathlib import Path
from distutils.util import strtobool
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import copy
from transformers import (
    AutoTokenizer,
    AutoConfig,
    set_seed,
    # get_scheduler,
)

from transformers import (
    DistilBertPreTrainedModel,
    DistilBertModel,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)

os.environ['TRANSFORMERS_CACHE']='/rscratch/tpang/kinshuk/cache'

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_path",
    type=str,
    default="/rscratch/tpang/kinshuk/RpMKin/dataset/english/new.csv",
    help="path for data file",
)
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
# ADD ARGUMENTS HERE

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# accelerate.utils.set_seed(args.seed)
set_seed(args.seed)  # transformers


class DistSpeech(DistilBertPreTrainedModel):
    """
    DistilBertForHateSpeech
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]

        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                # loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # loss_fct = CrossEntropyLoss()
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
                )

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )


def preprocess(input_text, tokenizer, max_length=128):
    return tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,  # Ensure sequences are truncated to the specified max_length
        return_attention_mask=True,
        return_tensors="pt",
    )


def get_train_val(args, df, batch_size=32, val_ratio=0.2, fraction=1):
    labels = df.label.values
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=val_ratio,
        shuffle=True,
        stratify=labels,
        random_state=args.seed,
    )

    text = df.text.values
    labels = df.label.values
    truncate_dataset = True

    if truncate_dataset:
        text = text[0 : int(len(text) * fraction)]
        labels = labels[0 : int(len(labels) * fraction)]

    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased", do_lower_case=True
    )

    token_id = []
    attention_masks = []

    max_len = 128

    # Tokenize all the samples
    for sample in text:
        encoding_dict = preprocess(sample, tokenizer, max_len)
        token_id.append(encoding_dict["input_ids"])
        attention_masks.append(encoding_dict["attention_mask"])

    # Convert the lists of tensors into a single tensor
    token_id = torch.stack([t.squeeze(0) for t in token_id])
    attention_masks = torch.stack([t.squeeze(0) for t in attention_masks])

    labels = torch.tensor(labels)

    train_set = TensorDataset(
        token_id[train_idx], attention_masks[train_idx], labels[train_idx]
    )

    val_set = TensorDataset(
        token_id[val_idx], attention_masks[val_idx], labels[val_idx]
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, cache_path='/rscratch/tpang/kinshuk/cache')
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, cache_path='/rscratch/tpang/kinshuk/cache')

    return train_loader, val_loader


def getOptimizer(model, lr, weight_decay):
    """
    optimizer
    """
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer


def getModel(model_name, num_labels):
    """
    model
    """
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, config=config)

    for param in model.parameters():  # type: ignore
        param.requires_grad = True
    return model


def val_loss(model, data_loader, device):
    """
    val loss
    """
    model.eval()
    total_loss = 0
    for batch in tqdm(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs[0]
            total_loss += loss.item()
    return total_loss / len(data_loader)


def train_loss(model, data_loader, optimizer, device):
    """
    train loss
    """
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(data_loader)


def tr_loss(model, train_dataloader, val_dataloader, optimizer, device):
    """
    train loss
    """
    num_steps = args.epochs * len(train_dataloader)

    progress_bar = tqdm(
        range(num_steps),
        disable=False
    )

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
    
        for batch in train_dataloader:
            input_ids, attention_mask, labels = batch

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss_fn = nn.CrossEntropyLoss()  # For multi-class classification
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.update(1)
        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {average_loss}")


def train(model, train_loader, val_loader, optimizer, device, epochs):
    """
    train
    """
    best_loss = np.inf
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss_ = train_loss(model, train_loader, optimizer, device)
        print(f"Train loss {train_loss_}")
        v_loss = val_loss(model, val_loader, device)
        print(f"Val loss {v_loss}")
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    """
    main
    """
    file_path = args.data_path
    # extract dataframe out of csv
    try:
        df = pd.read_csv(file_path)
        # Check if the 'text' and 'label' columns exist
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV file does not contain required 'text' and 'label' columns.")
    except ValueError as e:
        print(e)
        print("The available columns in the CSV are: ", df.columns)
        return  # Exit the program if the required columns are not found

    # If the columns exist, continue with the rest of the code
    df = df[["text", "label"]]
    print("data frame is:\n{}".format(df[:5]))
    print("Number of data samples:\n{}".format(len(df)))

    train_dataloader, val_dataloader = get_train_val(
        args, df, batch_size=32, val_ratio=0.2, fraction=1
    )
    print("Training data size: ", len(train_dataloader))
    print("Validation data size: ", len(val_dataloader))

    model = getModel("distilbert-base-uncased", 3)
    model.to(device)  # type: ignore
    print("Model loaded\n")
    print("Getting optimizer...")
    optimizer = getOptimizer(model, 1e-5, 0.01)
    print("Optimizer loaded\n")
    print("Training...")
    tr_loss(model, train_dataloader, val_dataloader, optimizer, device)
    print("Training complete\n")
    print("Saving model...")
    torch.save(model.state_dict(), "model.pth")  # type: ignore
    print("Model saved\n")
    print("Done")


if __name__ == "__main__":
    main()
