import torch
import os
import random

from Dataloader.DataLoader_NER import DataLoader
from DataUtils.Alphabet import CreateAlphabet
from DataUtils.Batch_Iterator import Iterators

def _convert_word2id(insts, operator):
    """
    :param insts:
    :param operator:
    :return:
    """
    # print(len(insts))
    # for index_inst, inst in enumerate(insts):
    for inst in insts:
        # copy with the word and pos
        for index in range(inst.words_size):
            word = inst.words[index]
            wordId = operator.word_alphabet.loadWord2idAndId2Word(word)
            # if wordID is None:
            if wordId == -1:
                wordId = operator.word_unkId
            inst.words_index.append(wordId)

            label = inst.labels[index]
            labelId = operator.label_alphabet.loadWord2idAndId2Word(label)
            inst.label_index.append(labelId)

            char_index = []
            for char in inst.chars[index]:
                charId = operator.char_alphabet.loadWord2idAndId2Word(char)
                if charId == -1:
                    charId = operator.char_unkId
                char_index.append(charId)
            inst.chars_index.append(char_index)

def preprocessing(config):
    print("Processing Data......")
    # read file
    data_loader = DataLoader(path=[config.train_file, config.dev_file, config.test_file], shuffle=True, config=config)
    train_data, dev_data, test_data = data_loader.dataLoader()
    print(
        "train sentence {}, dev sentence {}, test sentence {}.".format(len(train_data), len(dev_data), len(test_data)))


    # create the alphabet
    alphabet = None

    alphabet = CreateAlphabet(min_freq=config.min_freq, config=config)
    alphabet.build_vocab(train_data=train_data, dev_data=dev_data, test_data=test_data)

    alphabet_dict = {"alphabet": alphabet}
    if config.save_pkl:
        # pcl.save(obj=alphabet_dict, path=os.path.join(config.pkl_directory, config.pkl_alphabet))
        torch.save(obj=alphabet_dict, f=os.path.join(config.pkl_directory, config.pkl_alphabet))

    _convert_word2id(insts=train_data, operator=alphabet)
    _convert_word2id(insts=dev_data, operator=alphabet)
    _convert_word2id(insts=test_data, operator=alphabet)

    data_dict = {"train_data": train_data, "dev_data": dev_data, "test_data": test_data}
    if config.save_pkl:
        torch.save(obj=data_dict, f=os.path.join(config.pkl_directory, config.pkl_data))

    return train_data, dev_data, test_data, alphabet

def Create_Iterator(insts, batch_size, operator, epoch, config):

    lgth = len(insts)
    p = 0
    iter = Iterators(batch_size = batch_size, data = insts, operator = operator, config = config)
    if epoch == 1:
        while p < lgth:
            result = []
            for _ in range(min(batch_size, lgth - p)):
                result.append(insts[p])
                p+=1
            result = iter._Create_Each_Batch(insts=result, operator=operator)
            yield result
    else:
        for _ in range(int(lgth * epoch / batch_size)):
            result = []
            for _ in range(batch_size):
                if p >= lgth:
                    p = 0
                    random.shuffle(insts)
                result.append(insts[p])
                p+=1
            result = iter._Create_Each_Batch(insts=result, operator=operator)
            yield result

