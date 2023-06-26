from transformers import BertTokenizer
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
# from transformers import get_scheduler
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch.nn as nn
import os

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
weapon_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
victim_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
target_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)


weapon_chunks = []
with open('Colab/weapon_chunks.txt', 'r') as input_file:
  for line in input_file:
    line = line.strip()
    weapon_chunks.append(line)
input_file.close()

weapon_labels = []
with open('Colab/weapon_labels.txt', 'r') as input_file:
  for line in input_file:
    line = line.strip()
    weapon_labels.append(int(line))
input_file.close()

victim_chunks = []
with open('Colab/victim_chunks.txt', 'r') as input_file:
  for line in input_file:
    line = line.strip()
    victim_chunks.append(line)
input_file.close()

victim_labels = []
with open('victim_labels.txt', 'r') as input_file:
  for line in input_file:
    line = line.strip()
    victim_labels.append(int(line))
input_file.close()

target_chunks = []
with open('Colab/target_chunks.txt', 'r') as input_file:
  for line in input_file:
    line = line.strip()
    target_chunks.append(line)
input_file.close()

target_labels = []
with open('Colab/target_labels.txt', 'r') as input_file:
  for line in input_file:
    line = line.strip()
    target_labels.append(int(line))
input_file.close()


# Weapon and Perp

tokenized_train = tokenizer.batch_encode_plus(weapon_chunks, add_special_tokens=True, return_attention_mask=True, 
pad_to_max_length=True, max_length=512, return_tensors='pt')
input_ids_train = tokenized_train['input_ids']
attention_masks_train = tokenized_train['attention_mask']
labels_train = torch.tensor(weapon_labels)
dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)

train_dataloader = DataLoader(dataset=dataset_train, shuffle=True, batch_size=16)
num_epochs = 4
num_training_steps = num_epochs * len(train_dataloader)
optimizer = AdamW(params=weapon_model.parameters(), lr=5e-6)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_dataloader)*num_epochs)
progress_bar = tqdm(range(num_training_steps))
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
weapon_model.to(device)

# Tells the model that we are training the model
weapon_model.train()
# Loop through the epochs
for epoch in range(num_epochs):
    # Loop through the batches
    for batch in train_dataloader:
        # Get the batch
        batch = tuple(b.to(device) for b in batch)   
        weapon_model.zero_grad()
        # Compute the model output for the batch
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels':batch[2]}

        outputs = weapon_model(**inputs)
        # Loss computed by the model
        loss = outputs.loss
        # backpropagates the error to calculate gradients
        loss.backward()
        # Update the model weights
        optimizer.step()
        # Learning rate scheduler
        lr_scheduler.step()
        # Clear the gradients
        optimizer.zero_grad()
        # Update the progress bar
        progress_bar.update(1)

"""# Victim"""

tokenized_train = tokenizer.batch_encode_plus(victim_chunks, add_special_tokens=True, return_attention_mask=True, 
pad_to_max_length=True, max_length=512, return_tensors='pt')
input_ids_train = tokenized_train['input_ids']
attention_masks_train = tokenized_train['attention_mask']
labels_train = torch.tensor(victim_labels)
dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)

train_dataloader = DataLoader(dataset=dataset_train, shuffle=True, batch_size=16)
num_epochs = 5
num_training_steps = num_epochs * len(train_dataloader)
optimizer = AdamW(params=victim_model.parameters(), lr=5e-6)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_dataloader)*num_epochs)
progress_bar = tqdm(range(num_training_steps))
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
victim_model.to(device)

# Tells the model that we are training the model
victim_model.train()
# Loop through the epochs
for epoch in range(num_epochs):
    # Loop through the batches
    for batch in train_dataloader:
        # Get the batch
        batch = tuple(b.to(device) for b in batch)   
        victim_model.zero_grad()
        # Compute the model output for the batch
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels':batch[2]}

        outputs = victim_model(**inputs)
        # Loss computed by the model
        loss = outputs.loss
        # backpropagates the error to calculate gradients
        loss.backward()
        # Update the model weights
        optimizer.step()
        # Learning rate scheduler
        lr_scheduler.step()
        # Clear the gradients
        optimizer.zero_grad()
        # Update the progress bar
        progress_bar.update(1)

"""# Target"""

tokenized_train = tokenizer.batch_encode_plus(target_chunks, add_special_tokens=True, return_attention_mask=True, 
pad_to_max_length=True, max_length=512, return_tensors='pt')
input_ids_train = tokenized_train['input_ids']
attention_masks_train = tokenized_train['attention_mask']
labels_train = torch.tensor(target_labels)
dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)

train_dataloader = DataLoader(dataset=dataset_train, shuffle=True, batch_size=16)
num_epochs = 5
num_training_steps = num_epochs * len(train_dataloader)
optimizer = AdamW(params=target_model.parameters(), lr=5e-6)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_dataloader)*num_epochs)
progress_bar = tqdm(range(num_training_steps))
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
target_model.to(device)

# Tells the model that we are training the model
target_model.train()
# Loop through the epochs
for epoch in range(num_epochs):
    # Loop through the batches
    for batch in train_dataloader:
        # Get the batch
        batch = tuple(b.to(device) for b in batch)   
        target_model.zero_grad()
        # Compute the model output for the batch
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels':batch[2]}

        outputs = target_model(**inputs)
        # Loss computed by the model
        loss = outputs.loss
        # backpropagates the error to calculate gradients
        loss.backward()
        # Update the model weights
        optimizer.step()
        # Learning rate scheduler
        lr_scheduler.step()
        # Clear the gradients
        optimizer.zero_grad()
        # Update the progress bar
        progress_bar.update(1)


torch.save(weapon_model.state_dict(), 'bert_models/checkpoint_weapon_chunks.pth')
torch.save(target_model.state_dict(), 'bert_models/checkpoint_target_chunks.pth')
torch.save(victim_model.state_dict(), 'bert_models/checkpoint_victim_chunks.pth')

