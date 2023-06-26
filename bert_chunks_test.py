from transformers import BertTokenizer
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
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

weapon_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
state_dict = torch.load('bert_models/checkpoint_weapon_chunks.pth')
weapon_model.load_state_dict(state_dict)

state_dict = torch.load('bert_models/checkpoint_target_chunks.pth')
target_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
target_model.load_state_dict(state_dict)

state_dict = torch.load('bert_models/checkpoint_victim_chunks.pth')
victim_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
victim_model.load_state_dict(state_dict)


def get_labels_and_chunks(all_chunks, predicted_prob_all):
    predicted_labels = []
    predicted_chunks = [] 
    for i in range(len(all_chunks)):
        chunk = all_chunks[i].strip()
        predicted_chunks.append(chunk)
        all_prob = predicted_prob_all[i].cpu().numpy()
        label = np.argmax(all_prob)
        # print(chunk, all_prob, label, predictions_all[i])
        predicted_labels.append(label)

    return predicted_chunks, predicted_labels

def sort_labels_weapons(phrase, labels, weapons, perps):
    for i in range(len(phrase)):
        each_phrase = phrase[i].strip().upper()
        if labels[i] == 1:
            weapons.add(each_phrase)
        elif labels[i] == 2:
            perps.add(each_phrase)

def sort_labels_victims(phrase, labels, victims):
    for i in range(len(phrase)):
        each_phrase = phrase[i].strip().upper()
        if labels[i] == 1:
            victims.add(each_phrase)

def sort_labels_targets(phrase, labels, targets):
    for i in range(len(phrase)):
        each_phrase = phrase[i].strip().upper()
        if labels[i] == 3:
            targets.add(each_phrase)
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

#Get Test Data
test_chunks = dict()
incidents = dict()
name = ""
with open('Colab/test_chunks.txt', 'r') as input_file:
  for line in input_file:
    if line.strip() == "":
      continue
    if "ID:" in line:
      splits = line.split("\t")
      name = splits[1].strip()
      incident = splits[2].strip()
      test_chunks[name] = []
      incidents[name] = incident
    elif name != "":
      test_chunks[name].append(line.strip())

def bert_evaluate():

    for filename in test_chunks:
        weapons = set()
        targets = set()
        perps = set()
        victims = set()
        curr_file_chunks = test_chunks[filename]

        tokenized_test = tokenizer.batch_encode_plus(curr_file_chunks, add_special_tokens=True, return_attention_mask=True, 
        pad_to_max_length=True, max_length=512, return_tensors='pt')
        input_ids_test = tokenized_test['input_ids']
        attention_masks_test = tokenized_test['attention_mask']
        dataset_test = TensorDataset(input_ids_test, attention_masks_test)
        test_dataloader = DataLoader(dataset=dataset_test, shuffle=False, batch_size=16)

        logits_all_weapon = []
        logits_all_victim = []
        logits_all_target = []

        predicted_prob_all_weapon = []
        predicted_prob_all_victim = []
        predicted_prob_all_target = []

        weapon_model.eval()
        victim_model.eval()
        target_model.eval()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        for batch in test_dataloader:
            # Get the batch
            batch = tuple(b.to(device) for b in batch) 
            inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1]}
            # Disable the gradient calculation
            with torch.no_grad():
                # Compute the model output
                weapon_outputs = weapon_model(**inputs)
                victim_outputs = victim_model(**inputs)
                target_outputs = target_model(**inputs)

            # Get the logits
            logits_weapon = weapon_outputs.logits
            logits_victim = victim_outputs.logits
            logits_target = target_outputs.logits

            # Append the logits batch to the list
            logits_all_weapon.append(logits_weapon)
            logits_all_victim.append(logits_victim)
            logits_all_target.append(logits_target)

            # Get the predicted probabilities for the batch
            predicted_prob_weapon = torch.softmax(logits_weapon, dim=1)
            predicted_prob_victim = torch.softmax(logits_victim, dim=1)
            predicted_prob_target = torch.softmax(logits_target, dim=1)

            # Append the predicted probabilities for the batch to all the predicted probabilities
            predicted_prob_all_weapon.extend(predicted_prob_weapon)
            predicted_prob_all_victim.extend(predicted_prob_victim)
            predicted_prob_all_target.extend(predicted_prob_target)

            # # Get the predicted labels for the batch
            # predictions = torch.argmax(logits, dim=-1)
            # Append the predicted labels for the batch to all the predictions
            # predictions_all.extend(predictions)

        predicted_phrase_weapon, predicted_labels_weapon = get_labels_and_chunks(curr_file_chunks, predicted_prob_all_weapon)
        sort_labels_weapons(predicted_phrase_weapon, predicted_labels_weapon, weapons, perps)

        predicted_phrase_victim, predicted_labels_victim = get_labels_and_chunks(curr_file_chunks, predicted_prob_all_victim)
        sort_labels_victims(predicted_phrase_victim, predicted_labels_victim, victims)

        predicted_phrase_target, predicted_labels_target = get_labels_and_chunks(curr_file_chunks, predicted_prob_all_target)
        sort_labels_targets(predicted_phrase_target, predicted_labels_target, targets)
        if filename == "EXAMPLE":
            terminal_output(incidents[filename], weapons, perps, targets, victims)
        else:
            print_labels(filename, incidents[filename], weapons, perps, targets, victims)
