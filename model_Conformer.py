import os
import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from torch.optim import AdamW
import csv
from conformer import ConformerBlock

#  600分类
class MyDataset(Dataset):
    def __init__(self, data_dir, segment_len=128):
        self.data_dir = data_dir
        self.segment_len = segment_len

        mapping_path = Path(data_dir) / "mapping.json"
        mapping = json.load(mapping_path.open())
        self.speaker2id = mapping['speaker2id']  # map 映射speakid到一个id(number)

        # load metadata of training data
        metadata_path = Path(data_dir) / 'metadata.json'
        metadata = json.load(open(metadata_path))['speakers']  # 映射某个speak
        # id的所有话数据(feature_path, mel_len)

        # get total number of speaker
        self.speaker_num = len(metadata.keys())
        # print('s num', self.speaker_num)
        self.data = []
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances['feature_path'], self.speaker2id[speaker]])  # [feature_path, id]

    def __getitem__(self, index):
        feat_path, speaker = self.data[index]
        # lead preprocessed mel-spectrogram
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        # Sefment mel-spectrogram in to 'segment_len' frames.
        if len(mel) > self.segment_len:
            # randomly get the starting point of the segment
            start = random.randint(0, len(mel) - self.segment_len)
            mel = torch.FloatTensor(mel[start:start + self.segment_len])
        else:
            mel = torch.FloatTensor(mel)
        # turn the speaker id into long for computing loss later
        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker

    def __len__(self):
        return len(self.data)

    def get_speaker_number(self):
        return self.speaker_num


# metadata = json.load(open(Path('./data/Dataset') / 'metadata.json'))['speakers']
# print(metadata.keys())

def collate_batch(batch):
    # process features within a batch
    """
    collate a batch of data
    :param batch:
    :return:
    """
    mel, speaker = zip(*batch)
    # because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same
    mel = pad_sequence(mel, batch_first=True, padding_value=20)  # 使batch中每个句子长度相同，取最大的，用padding_value覆盖
    # mel: (batch size, length, 40)
    return mel, torch.FloatTensor(speaker).long()


def get_dataloader(data_dir, batch_size, n_workers):
    """Generate dataloader"""
    dataset = MyDataset(data_dir)
    speaker_num = dataset.get_speaker_number()
    # split dataset into training dataset and validation dataset
    trainlen = int(0.9 * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]
    trainset, validset = random_split(dataset, lengths)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=n_workers,
                              pin_memory=True, collate_fn=collate_batch)
    valid_loader = DataLoader(validset, batch_size=batch_size, num_workers=n_workers, drop_last=True, pin_memory=True
                              , collate_fn=collate_batch)

    return train_loader, valid_loader, speaker_num


# Model: Transformer
class Classifier(nn.Module):
    def __init__(self, d_model=220, n_spks=600, dropout=0.1):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        self.conformer_block = ConformerBlock(
            dim=d_model,
            dim_head=256,
            heads=1,  # set 1
            ff_mult=4,
            conv_expansion_factor=18,
            conv_kernel_size=41,
            attn_dropout=dropout,
            ff_dropout=dropout,
            conv_dropout=dropout
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=1
        )

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        # Project the the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential(
            # nn.Linear(d_model, d_model),
            # nn.ReLU(),
            nn.Linear(d_model, n_spks)
        )

    def forward(self, mels):
        """
        args:
          mels: (batch size, length, 40)
        return:
          out: (batch size, n_spks)
        """
        # out: (batch size, length, d_model)
        out = self.prenet(mels)
        # out: (length, batch size, d_model)
        out = out.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        # out = self.encoder_layer(out)
        # out = self.encoder(out)
        # out: (batch size, length, d_model)
        # out = self.conformer_block(out)
        out = self.conformer_block(out)
        out = out.transpose(0, 1)
        # mean pooling
        stats = out.mean(dim=1)

        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out


# Learning rate schedule
def get_cosine_shedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1
):
    def lr_lambda(current_step):
        # warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# Model fuction
def model_fn(batch, model, criterion, device):
    """Forward a batch through the model"""
    mels, labels = batch
    mels = mels.to(device)
    labels = labels.to(device)

    outs = model(mels)

    loss = criterion(outs, labels)

    # get the speaker id with hightest probability
    preds = outs.argmax(1)
    # compute accuracy
    accuracy = torch.mean((preds == labels).float())
    return loss, accuracy


def valid(dataloader, model, criterion, device):
    """validate on validation set."""
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc='Valid', unit=' uttr')

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()
        pbar.update(dataloader.batch_size)
        pbar.set_postfix(
            loss=f'{running_loss / (i + 1):.2f}',
            accuracy=f'{running_accuracy / (i + 1):.2f}'
        )
    pbar.close()
    model.train()
    return running_accuracy / len(dataloader)


def parse_args():
    """arguments"""
    config = {
        'data_dir': './Dataset',
        'save_path': 'model.ckpt',
        'batch_size': 32,
        'n_workers': 8,
        'valid_steps': 2000,
        'warmup_steps': 1000,
        'save_steps': 10000,
        'total_steps': 70000
    }
    return config


def main(data_dir, save_path, batch_size, n_workers, valid_steps, warmup_steps, total_steps, save_steps):
    """Main function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Info]: Use {device} now!')

    train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size, n_workers)
    train_iterator = iter(train_loader)
    print(f"[Info]: Finish loading data!", flush=True)

    model = Classifier(n_spks=speaker_num).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_shedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[Info]: Finish creating model!", flush=True)

    best_accuracy = -1.0
    best_state_dict = None

    pbar = tqdm(total=valid_steps, ncols=0, desc='Train', unit=' step')

    for step in range(total_steps):
        # get data
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, accuracy = model_fn(batch, model, criterion, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

        # update the model
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        # Log
        pbar.update()
        pbar.set_postfix(
            loss=f"{batch_loss:.2f}",
            accuracy=f"{batch_accuracy:.2f}",
            step=step + 1,
        )
        # Do validation
        if (step + 1) % valid_steps == 0:
            pbar.close()

            valid_accuracy = valid(valid_loader, model, criterion, device)

            # keep the best model
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        # Save the best model so far.
        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict, save_path)
            pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

    pbar.close()




class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        testdata_path = Path(data_dir) / "testdata.json"
        metadata = json.load(testdata_path.open())
        self.data_dir = data_dir
        self.data = metadata['utterances']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utterance = self.data[index]
        feat_path = utterance['feature_path']
        mel = torch.load(os.path.join(self.data_dir, feat_path))
        return feat_path, mel


def inference_collate_batch(batch):
    """Collate a batch of data."""
    feat_paths, mels = zip(*batch)

    return feat_paths, torch.stack(mels)


def parse_args2():
    """test arguments"""
    config = {
        'data_dir': './Dataset',
        'model_path': './model.ckpt',
        'output_path': './output.csv'
    }
    return config

def main2(data_dir, model_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info]: Use {device} now!")

    mapping_path = Path(data_dir) / "mapping.json"
    mapping = json.load(mapping_path.open())

    dataset = InferenceDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=8, collate_fn=inference_collate_batch)
    speaker_num = len(mapping["id2speaker"])
    model = Classifier(n_spks=speaker_num).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"[Info]: Finish creating model!", flush=True)
    results = [["Id", "Category"]]
    for feat_paths, mels in tqdm(dataloader):
        with torch.no_grad():
            mels = mels.to(device)
            outs = model(mels)
            preds = outs.argmax(1).cpu().numpy()
            for feat_path, pred in zip(feat_paths, preds):
                results.append([feat_path, mapping["id2speaker"][str(pred)]])

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)


if __name__ == '__main__':
    main(**parse_args())
    main2(**parse_args2())
