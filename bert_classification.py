import spacy
import os
import data
import truecase
from torch.utils.data import TensorDataset
import numpy as np
import bert_chunks_test
import bert_sent_test

nlp = spacy.load("en_core_web_lg")
directory = "Resources/TerrorismEventData/test-doc"

def get_incident_type(incident):
        if "kidnap" in incident or "kidnapped" in incident:
            return "KIDNAPPING"
        elif incident == "arson":
            return "ARSON"
        elif incident == "bomb" or incident == "bombing" or incident == "explosion":
            return "BOMBING"
        elif incident == "rob" or incident == "robbery":
            return "ROBBERY"
        else:
            return ""
        
incident_types = ["kidnap", "kidnapped", "kidnapping", "kidnaps", "arson", "bomb", "rob", "robbery", "bombing", "explosion"]

total_weapons = set()
with open("Resources/weapons.txt") as weapons_file:
        for line in weapons_file:
            total_weapons.add(line.strip())
weapons_file.close()

def remove_punctuation(sentence):
    new_sentence = ""
    for token in sentence:
        if not token.is_punct:
            new_sentence += token.text + " "    
    return new_sentence

# #insert token before and after chunks in sentence
def insert_tag_around_chunks(sentence, chunk):
    new_chunk = '[UNK] ' + chunk + ' [UNK]'
    new_sentence = sentence.replace(chunk, new_chunk, 1)
    return new_sentence
    
def get_file_output(directory):
    # Loading and preprocessing data
    test_data_sentences = data.get_sentences_for_directory(directory)
    incident = []
    for id in test_data_sentences:
        # if id != "DEV-MUC3-0779":
        #     continue
        incident = ""
        doc_sentences = test_data_sentences[id]
        all_chunks = []
        all_sentences = []
        for sentence in doc_sentences:
            verb = ""
            #INCIDENT + VERB
            for token in sentence: 
                if verb == "" and token.pos_ == "VERB":
                    verb = token.lemma_
                    # verb = verb.text
                lemma_word = token.lemma_
                lemma_word = lemma_word.lower().strip()
                #INCIDENT
                if incident == "" and lemma_word in incident_types:
                    incident = get_incident_type(lemma_word)
            if incident == "":
                incident = "ATTACK"
           
            #CHUNKS
            sentence = nlp(truecase.get_true_case(sentence.text))
            new_sentence = remove_punctuation(sentence)
            for chunks in sentence.noun_chunks:
                chunks = data.remove_determiners(chunks).strip()
                changed_sentence = insert_tag_around_chunks(new_sentence, chunks)
                all_sentences.append(changed_sentence)
                all_chunks.append(chunks)

        with open('Colab/test_chunks.txt', 'a') as test_file_chunks:
            test_file_chunks.write("ID:\t" + id + "\t" + incident + '\n')
            for chunk in all_chunks:
                test_file_chunks.write(chunk + '\n')

        with open('Colab/test_sentences.txt', 'a') as test_file_sents:
            test_file_sents.write("ID:\t" + id + "\t" + incident + '\n')
            for sent in all_sentences:
                test_file_sents.write(sent + '\n')

    bert_chunks_test.bert_evaluate()
    # bert_sent_test.bert_evaluate()
        
# # SINGLE FILE INPUT WITH TERMINAL OUTPUT
def get_terminal_output():
    print()
    print("PROCESSING DATA...")
    filename = "example.txt"
    data_sentences = []

    with open(filename, 'r') as train_file:
        doc = ''
        for line in train_file:
            line = line.replace('\n', ' ')
            if ((len(line)) == 0):
                continue
            line = line.lower()
            doc += line
        filename = filename.strip()
    doc = nlp(doc)
    data_sentences = list(doc.sents)

    incident = ""
    all_sentences = []
    all_chunks = []
    for sentence in data_sentences:
        verb = ""
        #INCIDENT + VERB
        for token in sentence: 
            if verb == "" and token.pos_ == "VERB":
                verb = token.lemma_
                # verb = verb.text
            lemma_word = token.lemma_
            lemma_word = lemma_word.lower()
            #INCIDENT
            if incident == "" and lemma_word in incident_types:
                incident = get_incident_type(lemma_word)
        incident = "ATTACK"
       
        #CHUNKS
        sentence = nlp(truecase.get_true_case(sentence.text))
        new_sentence = remove_punctuation(sentence)
        for chunks in sentence.noun_chunks:
            chunks = data.remove_determiners(chunks).strip()
            changed_sentence = insert_tag_around_chunks(new_sentence, chunks)
            all_sentences.append(changed_sentence)
            all_chunks.append(chunks)
    

    with open('Colab/test_chunks.txt', 'w') as test_file_chunks:
        test_file_chunks.write("ID:\t" + "EXAMPLE" + "\t" + incident)
        for chunk in all_chunks:
            test_file_chunks.write(chunk + '\n')

    with open('Colab/test_sentences.txt', 'w') as test_file_sents:
        test_file_sents.write("ID:\t" + "EXAMPLE" + "\t" + incident)
        for sent in all_sentences:
            test_file_sents.write(sent + '\n')

    bert_chunks_test.bert_evaluate()
    # bert_sent_test.bert_evaluate()

get_file_output("Resources/TerrorismEventData/test-doc")