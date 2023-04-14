import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import os
import tiktoken
from MultiHead import TransformerClassifierModel
from pathlib import Path
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import argparse
import time

# Run script to train model
""" 
python train_classifier.py --data AGNewsData --output_dir SavedModels --epochs 5 --batch_size 128
"""

args = argparse.ArgumentParser()

args.add_argument("--data", type = str, required=True)
args.add_argument("--output_dir", type = str, required=True)
args.add_argument("--epochs", type = int, required=True)
args.add_argument("--batch_size", type = int, default=32)

        
parser = args.parse_args()

def pad_tensor(source, length, padding_value):

    new_tensor = torch.zeros(size=(length,)).to(
        dtype=source.dtype, device=source.device).fill_(torch.tensor(padding_value))

    if source.shape[0] > length:
        new_tensor[:] = source[:length]
    else:
        new_tensor[:source.shape[0]] = source

    return new_tensor



def load_data(data_path):
    data = pd.read_csv(data_path)

    # concat the title and description into a new column text
    data["text"] = data["Title"] + " " + data["Description"]

    # drop the text and title columns
    data.drop(columns = ["Title", "Description"], axis = 1, inplace = True)
    # rename the column name Class Index to label
    data.rename(columns = {"Class Index": "label"}, inplace = True)

    return data



class DatasetProcessor(Dataset):
    def __init__(self, data_iter, encoder, seq_len, padding_value) -> None:
        super().__init__()

        seq_len = seq_len
        self.all_text = []
        self.all_labels = []

        for label, text in zip(data_iter["label"], data_iter["text"]):
            # in the dataset the labelling starts from 1 instead of 0
            # subtracts 1 from the label to begin from 0
            label = int(label) - 1
            # convert text to integers
            text = encoder.encode_ordinary(text)

            # convert labels to pytorch tensors
            label = torch.tensor(label, dtype=torch.long) 
            # converts texts to pytorch tensors
            text = torch.tensor(text, dtype=torch.long)

            # pad converted text tensors
            text = pad_tensor(text, seq_len, padding_value)

            # append the labels into a list
            self.all_labels.append(label)
            # append the texts into a list
            self.all_text.append(text)

        # concatenate the texts list
        self.all_text = torch.stack(self.all_text, dim=0)
        #concatenate the labels lists
        self.all_labels = torch.stack(self.all_labels, dim=0)

    @property
    def num_classes(self):
        number_of_classes = len(self.all_labels.unique())

        return number_of_classes

    def __len__(self):
        return self.all_text.shape[0]

    def __getitem__(self, index):

        text = self.all_text[index]
        label = self.all_labels[index]

        return text, label


def data_handler(batch_size, seq_len, vocab_size, text_encoder):
    train_iter = load_data(os.path.join(parser.data, "train.csv"))
    # load the test data
    test_iter = load_data(os.path.join(parser.data, "test.csv"))

    train_dataset = DatasetProcessor(
        train_iter,
        text_encoder,
        seq_len,
        padding_value=text_encoder.eot_token
    )

    num_classes = train_dataset.num_classes

    
    test_dataset = DatasetProcessor(
        test_iter,
        text_encoder,
        seq_len,
        padding_value=text_encoder.eot_token
    )


    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size
    )

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size
    )

    return train_dataloader, test_dataloader, num_classes


class Trainer():
    def __init__(self) -> None:
        # define the maximum sentence length for texts 
        self.seq_len = 256
        # use GPT pretrained tokenizer
        text_encoder = tiktoken.get_encoding("gpt2")
        # obtain the vocabulary size from the tokenizer
        self.vocab_size = text_encoder.n_vocab
        # batch size for training 
        self.batch_size = parser.batch_size
        # embedding dimension for the Transformer Classifier Model
        self.embedding_dim = 128
        # gradient clip value for clipping model gradients
        self.gradient_clip = 1.0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # learning rate for training
        self.lr = 1e-3

        #path to save trained models
        self.output_dir = parser.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.best_acc = 0
      
        
        # DataLoaders for training called from the data_handler function
        self.train_dataloader, self.test_dataloader, self.num_classes = data_handler(self.batch_size,
        self.seq_len, self.vocab_size, text_encoder)
        
        print("Dataset Classes: ", self.num_classes)

        self.model = TransformerClassifierModel(vocab_size = self.vocab_size, max_len = self.seq_len,
         embedding_dim = self.embedding_dim, num_classes=self.num_classes)
      
        # move model to GPU
        self.model = self.model.to(self.device)

        # optimizer for training
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

        self.loss_fn = nn.CrossEntropyLoss()
        # set the self.scalar for scaling the model's gradients to fp16
        self.scalar = torch.cuda.amp.GradScaler(enabled=True)


    def process_batch(self, text, label):

        text = text.to(self.device)
        label = label.to(self.device)

        # autocast casts the input tensors to fp16 for mixed precision training
        with torch.autocast(device_type = "cuda", dtype=torch.float16):
            predictions = self.model(text)

            loss = self.loss_fn(predictions, label)

        acc = (predictions.argmax(1) == label).sum().item()

        return loss, acc

    def save_best(self, epoch, acc):
        model_path = os.path.join(self.output_dir, f"best_epoch{epoch}_acc_{acc:.3f}.pth" )
        torch.save(self.model.state_dict(), model_path)

    def test(self, epoch):
        self.model.eval()
        print(f"Validating Epoch {epoch}")
        all_acc, total_count = 0, 0
        all_loss = []
        for text, label in tqdm(self.test_dataloader):

            with torch.no_grad():
                loss, acc = self.process_batch(text, label)

                all_acc += acc
                all_loss.append(loss.item())
                total_count += label.shape[0]

        acc = all_acc / total_count
        loss = sum(all_loss) / len(all_loss)
        if acc > self.best_acc:
            self.save_best(epoch, acc)         
            self.best_acc = acc
       
        return acc, loss

    def train(self):

        self.model.train()

        all_acc, total_count = 0, 0
        all_loss = []
        for epoch in range(parser.epochs):
            print(f"Training Epoch {epoch}")
            for text, label in tqdm(self.train_dataloader):

                self.optimizer.zero_grad()

                loss, acc = self.process_batch(text, label)

                """ 
                Scaling is applied to multiply model parameters' gradients with small magnitude by a scalar factor which prevents 
                gradients Underflow(a condition in which gradients with small magnitude vanish to zero)
                """
                self.scalar.scale(loss).backward()

                """
                We need to unscale the gradients before updating the model parameters to prevent the scaling factor
                used in scaling from interferring with model's paramters like learning rate. 
                self.scalar.unscale_(optimizer) unscales the gradients 
                """
                self.scalar.unscale_(self.optimizer)

                """ 
                Gradient clipping prevents gradients overflow(a condition in which large magnitude gradients 
                lead to exploding of gradients and causes unstable model training). It clips model gradients based on the gradient clipping value. 
                For example if the gradient clipping value set to value 1.0 it will clip any gradient whose value is larger than this.
                """
                if self.gradient_clip > 0.0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                # self.scalar.step skips optimizer.step() if the gradients contain infs or NaNs
                self.scalar.step(self.optimizer)
                # updates the model's parameters using self.scalar.update 
                self.scalar.update()

                all_acc += acc
                all_loss.append(loss.item())
                total_count += label.shape[0]

            acc = all_acc / total_count
            loss = sum(all_loss) / len(all_loss)
            test_acc, test_loss = self.test(epoch)

            print(f"Epoch: {epoch}, Train Acc: {acc:.3f}, Train Loss: {loss:.3f}, Test Acc: {test_acc:.3f}, Test Loss: {test_loss:.3f}")



if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
  


