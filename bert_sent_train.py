from transformers import AutoTokenizer, AutoModel
from transformers import BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.optim import AdamW
from tqdm.auto import tqdm
import os
# import evaluate
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

all_chunks = []
with open('Colab/all_chunks.txt', 'r') as input_file:
  for line in input_file:
    line = line.strip()
    all_chunks.append(line)
input_file.close()

all_labels = []
with open('Colab/all_labels.txt', 'r') as input_file:
  for line in input_file:
    line = line.strip()
    all_labels.append(int(line))
input_file.close()

all_sentences = []
with open('Colab/all_sentences.txt', 'r') as input_file:
  for line in input_file:
    line = line.strip()
    all_sentences.append(line)
input_file.close()

from typing_extensions import final
class PosModel(nn.Module):
    def __init__(self):
        super(PosModel, self).__init__()
        num_labels = 5
        self.base_model = AutoModel.from_pretrained('bert-base-uncased')
        self.base_model.to(device)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(1536, num_labels) # output features from bert is 768 and 2 is number of labels
        

    def forward(self, input_ids, attn_mask, indexes):
        outputs = self.base_model(input_ids, attention_mask=attn_mask)
        outputs = outputs['last_hidden_state']
        final_output = []
        # print(len(outputs))
        for i in range(len(outputs)):
          cls = outputs[i][0]
          fi = indexes[i][0]
          li = indexes[i][1]
          chunk_embeds = outputs[i][fi:li]
          # print(chunk_embeds)
          chunk_embeds = torch.tensor(chunk_embeds).to(device)
          #col averaging
          avg_chunk_embeds = torch.mean(chunk_embeds, dim=0).to(device)
          #catenating with cls
          temp_cat = torch.cat((cls, avg_chunk_embeds), dim=-1).to(device)
          final_output.append(temp_cat.detach().cpu().numpy())
        
        final_output = torch.Tensor(final_output).to(device)

        outputs = self.dropout(outputs[0])
        final_output = self.linear(final_output)
        return final_output


def get_chunk_indexes(input_id):
  tensor_ids = input_id
  sep_ids = (tensor_ids == 100).nonzero(as_tuple=True)
  sep_ids = sep_ids[0].cpu().numpy()
  first_ind = sep_ids[0] + 1
  last_ind = sep_ids[-1]
  return(first_ind, last_ind)

tokenized_train = tokenizer(all_sentences, add_special_tokens=True, return_attention_mask=True, 
                                              pad_to_max_length=True, max_length=512, return_tensors='pt').to(device)
input_ids_train = tokenized_train['input_ids']
attention_masks_train = tokenized_train['attention_mask']
labels_train = torch.tensor(all_labels)
dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
train_dataloader = DataLoader(dataset=dataset_train, shuffle=True, batch_size=16)

num_epochs = 4
num_training_steps = num_epochs * len(train_dataloader)


model = PosModel()
model.to(device)
criterion = nn.CrossEntropyLoss()

optimizer = AdamW(params=model.parameters(), lr=5e-6)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*num_epochs)
progress_bar = tqdm(range(num_training_steps))

# # Tells the model that we are training the model

model.train()
# # Loop through the epochs
for epoch in range(num_epochs):
    # Loop through the batches
    for batch in train_dataloader:
        # Get the batch
        batch = tuple(b.to(device) for b in batch)   
        model.zero_grad()
        # Compute the model output for the batch
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels':batch[2]}

        indexes = []
        for i in range(len(inputs['input_ids'])):
          input = input_ids_train[i]
          indexes.append(get_chunk_indexes(input))
        outputs = model.forward(inputs['input_ids'], inputs['attention_mask'], indexes)

        # loss = criterion(outputs, inputs['labels'])
        # backpropagates the error to calculate gradients
        pred = torch.softmax(outputs, dim=1)
        loss = criterion(pred, inputs['labels'])
        loss.backward()
        # Update the model weights
        optimizer.step()
        # Learning rate scheduler
        lr_scheduler.step()
        # Clear the gradients
        optimizer.zero_grad()
        # Update the progress bar
        progress_bar.update(1)

torch.save(model.state_dict(), 'bert_model/checkpoint_sentences.pth')

