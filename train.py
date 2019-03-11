import os
import numpy as np
import torch
import argparse
import datetime

import Config.config as configurable
from ner_model.al_api import get_al_model
from DataUtils.preprocess import preprocessing

def parse_argument():
    """
    :argument
    :return:
    """
    parser = argparse.ArgumentParser(description="NER & POS")
    parser.add_argument("-c", "--config", dest="config_file", type=str, default="./Config/config.cfg",
                        help="config path")
    parser.add_argument("--train", dest="train", action="store_true", default=True, help="train model")
    parser.add_argument("-p", "--process", dest="process", action="store_true", default=True, help="data process")
    parser.add_argument("-t", "--test", dest="test", action="store_true", default=False, help="test model")
    parser.add_argument("--t_model", dest="t_model", type=str, default=None, help="model for test")
    parser.add_argument("--t_data", dest="t_data", type=str, default=None,
                        help="data[train dev test None] for test model")
    parser.add_argument("--predict", dest="predict", action="store_true", default=False, help="predict model")
    args = parser.parse_args()
    # print(vars(args))
    config = configurable.Configurable(config_file=args.config_file)
    config.train = args.train
    config.process = args.process
    config.test = args.test
    config.t_model = args.t_model
    config.t_data = args.t_data
    config.predict = args.predict
    # config
    if config.test is True:
        config.train = False
    if config.t_data not in [None, "train", "dev", "test"]:
        print("\nUsage")
        parser.print_help()
        print("t_data : {}, not in [None, 'train', 'dev', 'test']".format(config.t_data))
        exit()
    print("***************************************")
    print("Data Process : {}".format(config.process))
    print("Train model : {}".format(config.train))
    print("Test model : {}".format(config.test))
    print("t_model : {}".format(config.t_model))
    print("t_data : {}".format(config.t_data))
    print("predict : {}".format(config.predict))
    print("***************************************")

    return config


def main():
    config = parse_argument()
    config.mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config.save_dir = os.path.join(config.save_direction, config.mulu)

    is_testing = False
    if is_testing == True:
        config.max_count = 100

    load_data_from_pkl = False
    if load_data_from_pkl is True:
        tmp = torch.load(f=os.path.join(config.pkl_directory, config.pkl_alphabet))
        vocab = tmp['alphabet']
        tmp = torch.load(f=os.path.join(config.pkl_directory, config.pkl_data))
        train = tmp['train_data']
        dev = tmp['dev_data']
        test = tmp['test_data']
    else:
        train, dev, test, vocab = preprocessing(config=config)

    model = get_al_model(config=config, vocab = vocab, seed=1)
    model.fit(train, vali=dev)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))

    main()

    print('The End')