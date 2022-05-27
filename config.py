from __future__ import print_function

import os
import sys
import logging

if sys.version_info[0] < 3:
    import ConfigParser as configparser
else:
    import configparser

config = configparser.SafeConfigParser()

get = config.get
config_dir = os.path.dirname(__file__)  #当前脚本运行的路径。

def parser_config(): #
    config_file = os.path.join(config_dir, "conf")

    if not os.path.exists(config_file):
        sys.stderr.write("Error: Unable to find the config file!\n")
        sys.exit(1)

    # parse the configuration
    global config
    config.readfp(open(config_file))


parser_config()

COMP = {
    "Permission": "permission",
    "Activity": "activity",
    "Service": "service",
    "Receiver": "receiver",
    "Provider": "provider",
    "Hardware": "hardware",
    "Intentfilter": 'intent-filter',
    "Android_API": "android_api",
    "Java_API": "java_api",
    "User_String": "const-string",
    "User_Class": "user_class",
    "User_Method": "user_method",
    "OpCode": "opcode",
    "Asset": "asset",
    "Notdefined": 'not_defined'
}

logging.basicConfig(level=logging.ERROR, filename=os.path.join(config_dir, "log"), filemode="w",
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
ErrorHandler = logging.StreamHandler()
ErrorHandler.setLevel(logging.ERROR)
ErrorHandler.setFormatter(logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'))

########################################################################
####################       以下参数需要自己配置       ######################
DREBIN_FEATURE_Param = {
    'feature_dimension': 4096,
    'output_dimension': 2,
}

USING_DATASET = {
    'malware': 'virustotal_2018_5M_17M',
    'benware': 'androzoo_benware_3M_17M'
}

DNN_3Layers_Param = {
    'random_seed': 23456,
    'hidden_neurons': [160, 160, 160],  # DNN has two hidden layers with each having 160 neurons
    'input_dim': DREBIN_FEATURE_Param['feature_dimension'],
    'output_dim': DREBIN_FEATURE_Param['output_dimension'],  # malicious vs. benign
    'n_epochs': 8,
    'batch_size': 128,
    'learning_rate': 0.001,
    'optimizer': 'adam'  # others are not supported
}

modelA_Param = {
    'model_name': 'model_A',
    'random_seed': 32456,
    'hidden_neurons': [100, 100, 100],  # DNN has two hidden layers with each having 160 neurons
    'input_dim': DREBIN_FEATURE_Param['feature_dimension'],
    'output_dim':  DREBIN_FEATURE_Param['output_dimension'],  # malicious vs. benign
    'n_epochs': 20,
    'batch_size': 128,
    'learning_rate': 0.001,
    'optimizer': 'adam'  # others are not supported
}
modelB_Param = {
    'model_name': 'model_B',
    'random_seed': 65234,
    'hidden_neurons': [120, 120,120],  # DNN has two hidden layers with each having 160 neurons
    'input_dim': DREBIN_FEATURE_Param['feature_dimension'],
    'output_dim': DREBIN_FEATURE_Param['output_dimension'],  # malicious vs. benign
    'n_epochs': 2,
    'batch_size': 128,
    'learning_rate': 0.001,
    'optimizer': 'adam'  # others are not supported
}
modelC_Param = {
    'model_name': 'model_C',
    'random_seed': 54231,
    'hidden_neurons': [160, 160,160],  # DNN has two hidden layers with each having 160 neurons
    'input_dim': DREBIN_FEATURE_Param['feature_dimension'],
    'output_dim': DREBIN_FEATURE_Param['output_dimension'],  # malicious vs. benign
    'n_epochs': 2,
    'batch_size': 128,
    'learning_rate': 0.001,
    'optimizer': 'adam'  # others are not supported
}
modelD_Param = {
    'model_name': 'model_D',
    'random_seed': 23236,
    'hidden_neurons': [200, 200,200],  # DNN has two hidden layers with each having 160 neurons
    'input_dim': DREBIN_FEATURE_Param['feature_dimension'],
    'output_dim': DREBIN_FEATURE_Param['output_dimension'],  # malicious vs. benign
    'n_epochs': 2,
    'batch_size': 128,
    'learning_rate': 0.001,
    'optimizer': 'adam'  # others are not supported
}
modelE_Param = {
    'model_name': 'model_E',
    'random_seed': 12348,
    'hidden_neurons': [140,140,140],  # DNN has two hidden layers with each having 160 neurons
    'input_dim': DREBIN_FEATURE_Param['feature_dimension'],
    'output_dim': DREBIN_FEATURE_Param['output_dimension'],  # malicious vs. benign
    'n_epochs': 2,
    'batch_size': 128,
    'learning_rate': 0.001,
    'optimizer': 'adam'  # others are not supported
}
modelF_Param = {
    'model_name': 'model_F',
    'random_seed': 87235,
    'hidden_neurons': [120, 120,120,120],  # DNN has two hidden layers with each having 160 neurons
    'input_dim': DREBIN_FEATURE_Param['feature_dimension'],
    'output_dim': DREBIN_FEATURE_Param['output_dimension'],  # malicious vs. benign
    'n_epochs': 2,
    'batch_size': 128,
    'learning_rate': 0.001,
    'optimizer': 'adam'  # others are not supported
}
modelH_Param = {
    'model_name': 'model_F',
    'random_seed': 87235,
    'hidden_neurons': [100, 100,100,100],  # DNN has two hidden layers with each having 160 neurons
    'input_dim': DREBIN_FEATURE_Param['feature_dimension'],
    'output_dim': DREBIN_FEATURE_Param['output_dimension'],  # malicious vs. benign
    'n_epochs': 2,
    'batch_size': 128,
    'learning_rate': 0.001,
    'optimizer': 'adam'  # others are not supported
}
modelTarget_Param = {
    'model_name': 'model_target',
    'random_seed': 56891,
    'hidden_neurons': [160, 160,160],  # DNN has two hidden layers with each having 160 neurons
    'input_dim': DREBIN_FEATURE_Param['feature_dimension'],
    'output_dim': DREBIN_FEATURE_Param['output_dimension'],  # malicious vs. benign
    'n_epochs': 20,
    'batch_size': 128,
    'learning_rate': 0.001,
    'optimizer': 'adam'  # others are not supported
}
modelSimulator_Param = {
    'model_name': 'model_simulator',
    'random_seed': 43423,
    'hidden_neurons': [160, 160, 160],  # DNN has two hidden layers with each having 160 neurons
    'input_dim': DREBIN_FEATURE_Param['feature_dimension'],
    'output_dim': DREBIN_FEATURE_Param['output_dimension'],  # malicious vs. benign
    'warmup_n_epochs': 5,
    'batch_size': 64,
    'warmup_learning_rate': 0.001,
    'optimizer': 'adam'  # others are not supported
}

model_160_Param = {
    'hidden_neurons': [160, 160, 160],  # DNN has two hidden layers with each having 160 neurons
    'input_dim': DREBIN_FEATURE_Param['feature_dimension'],
    'output_dim': DREBIN_FEATURE_Param['output_dimension'],  # malicious vs. benign
    'n_epochs': 2,
    'batch_size': 128,
    'learning_rate': 0.001,
    'optimizer': 'adam'  # others are not supported
}
"""
advtraining_models = ["model_A", "target_model", "model_B", "model_C", "model_D", "model_E", "model_F","model_0_160x3",
                      'model_1_160x3', 'model_2_160x3', 'model_3_160x3', 'model_4_160x3',
                      'model_5_160x3', 'model_6_160x3', 'model_7_160x3', 'model_8_160x3',
                      'model_9_160x3', ]
"""
advtraining_models = ["model_A", "target_model"]
model_Param_dict = {
    "model_A": modelA_Param,
    "model_B": modelB_Param,
    "model_C": modelC_Param,
    "model_D": modelD_Param,
    "model_E": modelE_Param,
    "model_F": modelF_Param,
    "model_H": modelH_Param,
    'target_model': modelTarget_Param,
    'model_0_160x3': model_160_Param,
    'model_1_160x3': model_160_Param,
    'model_2_160x3': model_160_Param,
    'model_3_160x3': model_160_Param,
    'model_4_160x3': model_160_Param,
    'model_5_160x3': model_160_Param,
    'model_6_160x3': model_160_Param,
    'model_7_160x3': model_160_Param,
    'model_8_160x3': model_160_Param,
    'model_9_160x3': model_160_Param,
}

advtraining_methods = ["pgdl2", "pgdl1", "pgd_linfinity"]