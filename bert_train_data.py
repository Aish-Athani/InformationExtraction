from transformers import Trainer
from sklearn import svm
import spacy
import data
import truecase

nlp = spacy.load("en_core_web_lg")
ans_directory = "Resources/TerrorismEventData/train-ans"
output_model = './models/bert_weapon.pth'
path = 'bert-base-uncased'

def remove_punctuation(sentence):
    new_sentence = ""
    for token in sentence:
        if not token.is_punct:
            new_sentence += token.text + " "    
    return new_sentence

def split_sentences(sentence):
    sentence_list = sentence.split()
    for val in sentence_list:
        if val.strip() == "":
            sentence_list.remove(val)
    return sentence_list

# #insert token before and after chunks in sentence
def insert_tag_around_chunks(sentence, chunk):
    new_chunk = '[UNK] ' + chunk + ' [UNK]'
    new_sentence = sentence.replace(chunk, new_chunk, 1)
    return new_sentence
    

def get_answerkey_values(train_filename_set):
    weapon_answers = dict()
    perp_answers = dict()
    target_answers = dict()
    victim_answers = dict()

    for filename in train_filename_set:
        filename = filename.replace(" ", "")
        with open(ans_directory + "/" + filename + ".anskey", 'r') as train_ans_file:
            curr_weapon = set()
            curr_perp = set()
            curr_target = set()
            curr_victim = set()

            is_weapon = False
            is_perp = False
            is_target = False
            is_victim = False

            for line in train_ans_file:
                if "WEAPON:" in line:
                    line = line.split(':')[1].strip()
                    curr_weapon = set()
                    if len(line) == 0:
                        continue
                    elif len(line) == 1 and '-' in line:
                        continue
                    elif '/' in line:
                        for each_weapon in line.split(' / '):
                            if (len(each_weapon.strip()) == 0):
                                    continue
                            curr_weapon.add(each_weapon.strip())
                        is_weapon = True
                    else:
                        curr_weapon.add(line.strip())
                        is_weapon = True
                elif "PERP " in line:
                    is_weapon = False
                    line = line.split(':')[1].strip()
                    if len(line) == 0:
                        continue
                    elif len(line) == 1 and '-' in line:
                        continue
                    elif '/' in line:
                        for each_perp in line.split('/'):
                            if (len(each_perp.strip()) == 0):
                                continue
                            curr_perp.add(each_perp.strip())
                        is_perp = True
                    else:
                        if (len(line.strip()) != 0):
                            curr_perp.add(line.strip())
                            is_perp = True
                if "TARGET:" in line:
                    is_perp = False  
                    line = line.split(':')[1].strip()
                    curr_target = set()
                    if len(line) == 0:
                        continue
                    elif len(line) == 1 and '-' in line:
                        continue
                    elif '/' in line:
                        for each_target in line.split('/'):
                            if (len(each_target.strip()) == 0):
                                continue
                            curr_target.add(each_target.strip())
                        is_target = True
                    else:
                        curr_target.add(line.strip())
                        is_target = True 
                elif "VICTIM:" in line:
                    is_target = False
                    line = line.split(':')[1].strip()
                    curr_victim = set()
                    if len(line) == 0:
                        continue
                    elif len(line) == 1 and '-' in line:
                        continue
                    elif '/' in line:
                        for each_victim in line.split('/'):
                            if (len(each_victim.strip()) == 0):
                                continue
                            curr_victim.add(each_victim.strip())
                        is_victim = True
                    else:
                        if (len(line.strip()) != 0):
                            curr_victim.add(line.strip())
                            is_victim = True
                elif is_weapon:
                    if '/' in line:
                        for each_weapon in line.split(' / '):
                            if (len(each_weapon.strip()) == 0):
                                    continue
                            curr_weapon.add(each_weapon.strip())
                        is_weapon = True
                    else:
                        curr_weapon.add(line.strip())
                        is_weapon = True
                elif is_perp:
                    if '/' in line:
                        for each_perp in line.split('/'):
                            if (len(each_perp.strip()) == 0):
                                continue
                            curr_perp.add(each_perp.strip())
                    else:
                        if (len(line.strip()) != 0):
                            curr_perp.add(line.strip())
                elif is_target:
                    if '/' in line:
                        for each_target in line.split('/'):
                            if (len(each_target.strip()) == 0):
                                continue
                            curr_target.add(each_target.strip())
                        is_target = True
                    else:
                        curr_target.add(line.strip())
                        is_target = True
                elif is_victim:
                    if '/' in line:
                        for each_victim in line.split('/'):
                            if (len(each_victim.strip()) == 0):
                                continue
                            curr_victim.add(each_victim.strip())
                    else:
                        if (len(line.strip()) != 0):
                            curr_victim.add(line.strip())

        if (len(curr_weapon) > 0):
            weapon_answers[filename] = curr_weapon
        if (len(curr_perp) > 0):
            perp_answers[filename] = curr_perp
        if (len(curr_target) > 0):
            target_answers[filename] = curr_target
        if (len(curr_victim) > 0):
            victim_answers[filename] = curr_victim
        train_ans_file.close()
    return weapon_answers, perp_answers, target_answers, victim_answers

def train_data_extraction():
    train_filename_set = data.get_all_train_filenames()
    train_data_sentences = data.get_train_data_sentences()
    weapon_answers, perp_answers, target_answers, victim_answers = get_answerkey_values(train_filename_set)
    all_labels = []
    all_chunks = []
    all_sentences = []
    count_pos = 0
    # temp_sent = "text a bomb exploded in the bathroom"
    for id in train_data_sentences:
        doc_sentence = train_data_sentences[id]
        #No weapons in this article
        # if id not in weapon_answers.keys() and id not in perp_answers.keys():
        #     continue
        # go through each sentence
        for sentence in doc_sentence:
            contains_label = False
            if id in weapon_answers:
                for each_weapon in weapon_answers[id]:
                    if each_weapon in sentence.text:
                        contains_label = True
                        break
            if not contains_label and id in perp_answers:
                for each_perp in perp_answers[id]:
                    if each_perp in sentence.text:
                        contains_label = True
                        break
            #if sentence contains weapon create the context vector
            if contains_label:
                #Get verb
                sentence = truecase.get_true_case(sentence.text)
                sentence = nlp(sentence)
                new_sentence = remove_punctuation(sentence)
                # if temp_sent not in new_sentence:
                #     continue
                for np_chunks in sentence.noun_chunks:
                    chunk = data.remove_determiners(np_chunks).strip()
                    curr_upper = chunk.upper()
                    if id in weapon_answers and curr_upper in weapon_answers[id]:
                        changed_sentence = insert_tag_around_chunks(new_sentence, chunk)
                        all_sentences.append(changed_sentence)
                        all_chunks.append(chunk)
                        all_labels.append(1)
                        count_pos = 0
                        if count_pos == 4:
                            count_pos = 0
                    if id in victim_answers and curr_upper in victim_answers[id]:
                        changed_sentence = insert_tag_around_chunks(new_sentence, chunk)
                        all_sentences.append(changed_sentence)
                        all_chunks.append(chunk)
                        all_labels.append(2)
                        count_pos = 0
                    if id in target_answers and curr_upper in target_answers[id]:
                        changed_sentence = insert_tag_around_chunks(new_sentence, chunk)
                        all_sentences.append(changed_sentence)
                        all_chunks.append(chunk)
                        all_labels.append(3)
                        count_pos = 0
                    elif id in perp_answers and curr_upper in perp_answers[id]:
                        changed_sentence = insert_tag_around_chunks(new_sentence, chunk)
                        all_sentences.append(changed_sentence)
                        all_chunks.append(chunk)
                        all_labels.append(4)
                        # count_pos = 0
                    else: 
                        if count_pos < 7:
                            changed_sentence = insert_tag_around_chunks(new_sentence, chunk)
                            if "[UNK]" in changed_sentence:
                                all_sentences.append(changed_sentence)
                                all_chunks.append(chunk)
                                all_labels.append(0)
                                count_pos += 1

    
    with open('Colab/all_chunks.txt', 'w') as answer_file:
        for chunk in all_chunks:
            answer_file.write(chunk + '\n')
    with open('Colab/all_labels.txt', 'w') as file:
        for label in all_labels:
            file.write(str(label) + '\n')
    with open('Colab/all_sentences.txt', 'w') as file:
        for sentence in all_sentences:
            file.write(sentence + '\n')
            
train_data_extraction()
