from config import config
import os
import numpy as np
def reconstruct_checkpoint_file(checkpoint_dir):
    with open(checkpoint_dir, 'r') as f:
        model_checkpoint_path = f.readline()
    key_value = "checkpoint-"
    split_list = model_checkpoint_path.split(key_value)
    after = split_list[-1]
    model_checkpoint_path = "model_checkpoint_path: \"" + checkpoint_dir+"-"+after
    all_model_checkpoint_paths = "all_model_checkpoint_paths: \"" + checkpoint_dir+"-"+after
    with open(checkpoint_dir, 'w') as f:
        f.write(model_checkpoint_path)
        f.write(all_model_checkpoint_paths)

if __name__ == "__main__":
    advtraining_method = "pgdl1"
    malware_dataset_name = "virustotal_2018_5M_17M"
    benware_dataset_name = "androzoo_benware_3M_17M"
    exp_name = "3_4L_nobalanced_nobenwaretrain_15_epochs_4096_" + malware_dataset_name + "_AND_" + benware_dataset_name
    adv_train_root = config.get("advtraining.drebin", "advtraining_drebin_root") + "/" + advtraining_method + \
                     "_" + exp_name
    all_model = os.listdir(adv_train_root)
    for model in all_model:
        advtraining_drebin_pgdl2_modelx = adv_train_root + "/" + model
        advx = os.listdir(advtraining_drebin_pgdl2_modelx)
        advtraining_drebin_pgdl2_modelx_advx = [advtraining_drebin_pgdl2_modelx + "/" + advx[i] for i in
                                                range(len(advx))]
        for _dir in advtraining_drebin_pgdl2_modelx_advx:
            path = _dir + "/checkpoint"
            if os.path.exists(_dir) is False:
                continue
            reconstruct_checkpoint_file(path)
