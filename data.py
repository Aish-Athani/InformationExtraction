from sklearn import svm
import spacy
import os

nlp = spacy.load("en_core_web_lg")
directory = "Resources/TerrorismEventData/train-doc"
directory_test = "Resources/TerrorismEventData/test-doc"

def get_all_train_filenames():
    train_filename_set = set()
    
    for filename in os.listdir(directory):
        if("." in filename):
            continue
        train_filename_set.add(filename)
    return train_filename_set


def get_all_test_filenames():
    test_filename_set = set()
    
    for filename in os.listdir(directory_test):
        if("." in filename):
            continue
        test_filename_set.add(filename)
    return 

def get_all_filenames(directory_curr):
    filename_set = set()
    
    for filename in os.listdir(directory_curr):
        if("." in filename):
            continue
        filename_set.add(filename)
    return filename_set

def get_train_data_sentences():
    train_data_sentences = dict()
    train_filename_set = get_all_train_filenames()
    for filename in train_filename_set:
        with open(directory + "/" + filename, 'r') as train_file:
            doc = ''
            for line in train_file:
                line = line.replace('\n', ' ')
                if ((len(line)) == 0):
                    continue
                doc += line
            filename = filename.strip()
        doc = nlp(doc)
        train_data_sentences[filename] = list(doc.sents)
        train_file.close()
    return train_data_sentences

def get_train_data():
    train_data = dict()
    train_filename_set = get_all_train_filenames()
    for filename in train_filename_set:
        with open(directory + "/" + filename, 'r') as train_file:
            id = ''
            doc = ''
            for line in train_file:
                line = line.replace('\n', ' ')
                if ((len(line)) == 0):
                    continue
                line = line.lower()
                doc += line
            filename = filename.strip()
        train_data[filename] = doc
        doc = nlp(doc)
    return train_data

def get_test_data_sentences():
    test_data_sentences = dict()
    test_filename_set = get_all_test_filenames()
    for filename in test_filename_set:
        with open(directory_test + "/" + filename, 'r') as test_file:
            doc = ''
            for line in test_file:
                line = line.replace('\n', ' ')
                if ((len(line)) == 0):
                    continue
                doc += line
            filename = filename.strip()
        doc = nlp(doc)
        test_data_sentences[filename] = list(doc.sents)
        test_file.close()
    return test_data_sentences

def get_sentences_for_directory(directory):
    test_data_sentences = dict()
    test_filename_set = get_all_filenames(directory)

    for filename in test_filename_set:
        with open(directory_test + "/" + filename, 'r') as test_file:
            doc = ''
            for line in test_file:
                line = line.replace('\n', ' ')
                if ((len(line)) == 0):
                    continue
                doc += line
            filename = filename.strip()
        doc = nlp(doc)
        test_data_sentences[filename] = list(doc.sents)
        test_file.close()
    return test_data_sentences

#remove stopwords and punctuation
def get_cleaned_data(sentence):
    new_sentence = ""
    for token in sentence:
        if not token.is_stop and not token.is_punct:
            new_sentence += token.text + " "
    return new_sentence


#remove preposition 
def remove_determiners(sentence):
    new_sentence = ""
    for token in sentence:
        if token.pos_ is not "DET":
            new_sentence += token.text + " "
    return new_sentence