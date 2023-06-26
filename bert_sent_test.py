from transformers import AutoTokenizer, AutoModel
from transformers import BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
# from transformers import get_scheduler
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.optim import AdamW
from tqdm.auto import tqdm
import os

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from typing_extensions import final
class PosModel(nn.Module):
    def __init__(self):
        super(PosModel, self).__init__()
        num_labels = 4
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

model = PosModel()
state_dict = torch.load('bert_models/checkpoint_sentences.pth')
model.load_state_dict(state_dict)

def get_labels_and_chunks(all_sentences, predicted_prob_all):
    predicted_labels = []
    predicted_chunks = [] 

    for i in range(len(all_sentences)):
        sentence = all_sentences[i].strip().split("[UNK]")
        chunk = sentence[1].strip()
        predicted_chunks.append(chunk)
        all_prob = predicted_prob_all[i].cpu().numpy()
        label = np.argmax(all_prob)
        # print(chunk, all_prob, label, predictions_all[i])
        predicted_labels.append(label)

    return predicted_chunks, predicted_labels


def sort_labels(phrase, labels, weapons, perps, victims, targets):
    for i in range(len(phrase)):
        each_phrase = phrase[i].strip().upper()
        if labels[i] == 1:
            weapons.add(each_phrase)
            print("Weapon: ", each_phrase)
        elif labels[i] == 2:
            victims.add(each_phrase)
            print("Victims: ", each_phrase)
        elif labels[i] == 3:
            targets.add(each_phrase)
            print("Target: ", each_phrase)
        elif labels[i] == 4:
            perps.add(each_phrase)
            print("Perp: ", each_phrase)

def terminal_output(name, incident, weapons, perps, targets, victims):
    print("ID:\t" + name + "\nINCIDENT:\t" + incident)
    if len(weapons) == 0:
        print("WEAPON: - ")
    else: 
        print("WEAPON: ")
        for each_weapon in weapons:
            each_weapon = each_weapon.upper()
            print('\t' + each_weapon)
    if len(perps) == 0:
        print("PERP: - ")
    else: 
        print("PERP:")
        for each_perp in perps:
            each_perp = each_perp.upper()
            print('\t' + each_perp + '\n')
    if len(victims) == 0:
        print("VICTIM: -")
    else: 
        print("VICTIM:" )
        for each_victim in victims:
            print('\t' + each_victim)
    if len(targets) == 0:
        print("TARGET: -")
    else: 
        print("TARGET: ")
        for each_target in targets:
            each_target = each_target.upper()
            print('\t' + each_target)

def print_labels(name, incident, weapons, perps, targets, victims):
  filepath = os.path.join("Responses", name)
  with open(filepath, 'w') as answer_file:
    answer_file.write("ID:\t" + name + "\nINCIDENT:\t" + incident)
    answer_file.write("\nWEAPON:")
    if len(weapons) == 0:
        answer_file.write("\t-\n")
    else: 
        for each_weapon in weapons:
            each_weapon = each_weapon.upper()
            answer_file.write('\t' + each_weapon + '\n')
    answer_file.write("PERP:")
    if len(perps) == 0:
        answer_file.write("\t-\n")
    else: 
        for each_perp in perps:
            each_perp = each_perp.upper()
            answer_file.write('\t' + each_perp + '\n')

    answer_file.write("VICTIM:")
    if len(victims) == 0:
        answer_file.write("\t-\n")
    else: 
        for each_victim in victims:
            answer_file.write('\t' + each_victim + '\n')
    answer_file.write("TARGET:")
    if len(targets) == 0:
        answer_file.write("\t-")
    else: 
        for each_target in targets:
            each_target = each_target.upper()
            answer_file.write('\t' + each_target + '\n')
  answer_file.close()

test_chunks = dict()
name = ""
with open('test_chunks.txt', 'r') as input_file:
  for line in input_file:
    if line.strip() == "":
      continue
    if "ID:" in line:
      splits = line.split("\t")
      name = splits[1].strip()
      # incident = splits[2].strip()
      test_chunks[name] = []
    elif name != "":
      test_chunks[name].append(line.strip())

test_sentences = dict()
incidents = dict()
with open('test_sentences.txt', 'r') as input_file:
  for line in input_file:
    if line.strip() == "":
      continue
    if "ID:" in line:
      splits = line.split("\t")
      name = splits[1].strip()
      incident = splits[2].strip()
      # incident = "ATTACK"
      incidents[name] = incident
      test_sentences[name] = []
    elif name != "":
      if "[UNK]" not in line:
        continue
      test_sentences[name].append(line.strip())

def get_chunk_indexes(input_id):
  tensor_ids = input_id
  sep_ids = (tensor_ids == 100).nonzero(as_tuple=True)
  sep_ids = sep_ids[0].cpu().numpy()
  first_ind = sep_ids[0] + 1
  last_ind = sep_ids[-1]
  return(first_ind, last_ind)

   
def bert_evaluate():
   for filename in test_sentences:
    weapons = set()
    targets = set()
    perps = set()
    victims = set()

    curr_file_chunks = test_chunks[filename]
    curr_file_sentences = test_sentences[filename]
    # if filename != "DEV-MUC3-0766":
    #   continue

    tokenized_test = tokenizer.batch_encode_plus(curr_file_sentences, add_special_tokens=True, return_attention_mask=True, 
    pad_to_max_length=True, max_length=512, return_tensors='pt')
    input_ids_test = tokenized_test['input_ids']
    attention_masks_test = tokenized_test['attention_mask']
    dataset_test = TensorDataset(input_ids_test, attention_masks_test)
    test_dataloader = DataLoader(dataset=dataset_test, shuffle=False, batch_size=16)

    logits_all = []
    predicted_prob_all = []
    predictions_all = []
    model.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for batch in test_dataloader:
        # Get the batch
        batch = tuple(b.to(device) for b in batch) 
        inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1]}
        # Disable the gradient calculation
        with torch.no_grad():
        # Compute the model output
            indexes = []
            for i in range(len(inputs['input_ids'])):
                input = inputs['input_ids'][i]
                indexes.append(get_chunk_indexes(input))
            outputs = model.forward(inputs['input_ids'], inputs['attention_mask'], indexes)
            # outputs = model(**inputs)
        # Get the logits
        # logits = outputs.logits
        # print(logits.shape)
        # Append the logits batch to the list

        logits_all.append(outputs)
        # Get the predicted probabilities for the batch
        predicted_prob = torch.softmax(outputs, dim=1)
        # Append the predicted probabilities for the batch to all the predicted probabilities
        predicted_prob_all.extend(predicted_prob)

    predicted_phrase, predicted_labels = get_labels_and_chunks(curr_file_sentences, predicted_prob_all)
    sort_labels(predicted_phrase, predicted_labels, weapons, perps, victims, targets)
    if filename == "EXAMPLE":
      terminal_output(incidents[filename], weapons, perps, targets, victims)
    else:
      print_labels(filename, incidents[filename], weapons, perps, targets, victims)
