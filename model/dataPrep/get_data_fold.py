# import libraries
import pandas as pd

# Functions
def data_read(logger, root_dir):
    ## data is already shuffled with random_state=42
    all_train_data = pd.read_json(root_dir+'/Data/alltrainDF.json', orient = 'records')
    train_data = pd.read_json(root_dir+'/Data/trainDF.json', orient = 'records')
    dev_data = pd.read_json(root_dir+'/Data/devDF.json', orient = 'records')
    test_data = pd.read_json(root_dir+'/Data/testDF.json', orient = 'records')
    logger.info("all_train_data.shape {}".format(all_train_data.shape))
    logger.info("train_data.shape {}".format(train_data.shape))
    logger.info("dev_data.shape {}".format(dev_data.shape))
    logger.info("test_data.shape {}".format(test_data.shape))
    
    return all_train_data, train_data, dev_data, test_data