import matplotlib.pyplot as plt
import seaborn as sns
from tools.file_operation import read_pickle
from tools.feature_reverser import DrebinFeatureReverse
from tools import utils
from config import config
import numpy as np

input_dim = 4096
malware_dataset_name = "virustotal_2018_5M_17M"
benware_dataset_name = "androzoo_benware_3M_17M"
def load_data(software_type):
    if software_type is "malware":
        dataset_name = malware_dataset_name
    elif software_type is "benware":
        dataset_name = benware_dataset_name
    else:
        raise ValueError("wrong software type")

    feature_reverser = DrebinFeatureReverse(feature_mp='binary')  # drebin
    insertion_perm_array, removal_perm_array = feature_reverser.get_mod_array()
    feature_vectors = read_pickle(config.get(dataset_name, "sample_vectors"))
    if software_type is "malware":
        labels = np.ones(len(feature_vectors))
    elif software_type is "benware":
        labels = np.zeros(len(feature_vectors))
    else:
        raise ValueError("wrong software type")

    return feature_vectors, labels, insertion_perm_array, removal_perm_array, feature_reverser.normalizer

def load_search_domain(software_type):
    malware_feature_vectors, _, insertion_perm_array, removal_perm_array, normalizer = load_data(software_type)
    clip_min, clip_max = utils.get_min_max_bound(normalizer=normalizer)
    scaled_clip_min = utils.normalize_transform(np.reshape(clip_min, (1, -1)), normalizer=normalizer)
    scaled_clip_max = utils.normalize_transform(np.reshape(clip_max, (1, -1)), normalizer=normalizer)
    scaled_max_extended_var = np.maximum(
        np.multiply(scaled_clip_max,
                    insertion_perm_array) +  # upper bound for positions allowing perturbations
        np.multiply(scaled_clip_min, 1. - insertion_perm_array),
        malware_feature_vectors  # upper bound for positions no perturbations allowed
    )
    scaled_min_extended_var = np.minimum(
        np.multiply(scaled_clip_min, removal_perm_array) +
        np.multiply(scaled_clip_max, 1. - removal_perm_array),
        malware_feature_vectors
    )

    increase_search_domain = np.array((malware_feature_vectors < scaled_max_extended_var)).astype(np.int32)
    decrease_search_domain = np.array((malware_feature_vectors > scaled_min_extended_var)).astype(np.int32)
    increase_decrease_search_domain = \
        np.array((malware_feature_vectors < scaled_max_extended_var) | (malware_feature_vectors > scaled_min_extended_var)).astype(np.int32)
    visual_increase_search_domain = np.reshape(np.sum(increase_search_domain.astype(np.int32), axis=0), (100, 100))
    visual_decrease_search_domain = np.reshape(np.sum(decrease_search_domain.astype(np.int32), axis=0), (100, 100))
    visual_increase_decrease_search_domain = np.reshape(np.sum(increase_decrease_search_domain.astype(np.int32), axis=0), (100,100))
    visual_malware_feature_vectors = np.reshape(np.sum(malware_feature_vectors, axis=0),(100,100))

    return visual_increase_decrease_search_domain, visual_increase_search_domain, visual_decrease_search_domain, \
           visual_malware_feature_vectors

def load_attacked_samples(adv_samples_path, ori_samples_path):
    x_adv = utils.read_pickle(adv_samples_path).astype(np.int32)
    x_ori = utils.read_pickle(ori_samples_path).astype(np.int32)
    visual_diff = np.sum(np.abs(x_adv-x_ori), axis=0)
    visual_diff = np.reshape(visual_diff, (100, 100))
    return visual_diff

def visualization():
    #TwoL_adv_samples_path = "E:/jupyter_official/adv_samples/jsma/adv_TwoL_jsma_nomask_withFI"
    #TwoL_ori_samples_path = "E:/jupyter_official/adv_samples/jsma/ori_TwoL_jsma_nomask_withFI"
    #adv_samples_path = "E:/jupyter_official/adv_training/model_A/adv_samples/jsma/adv8/adv"
    #ori_samples_path = "E:/jupyter_official/adv_training/model_A/adv_samples/jsma/adv8/ori"

    #tar_adv_samples_path = "E:/jupyter_official/adv_training/target_model/adv_samples/jsma/adv0/adv"
    #tar_ori_samples_path = "E:/jupyter_official/adv_training/target_model/adv_samples/jsma/adv0/ori"
    #visual_InDe, visual_In, Visual_De, Visual_Samples = load_search_domain(software_type="malware")
    #visual_diff_1 = load_attacked_samples(adv_samples_path, ori_samples_path)
    #visual_diff_2 = load_attacked_samples(tar_adv_samples_path, tar_ori_samples_path)
    #visual_diff = visual_diff_1 + visual_diff_2
    #print(visual_diff_1.shape)
    #visual_InDe, visual_In, Visual_De, Visual_Samples = load_search_domain("all")
    feature_vectors, labels, insertion_perm_array, removal_perm_array, normalizer = load_data("malware")
    f, ax1 = plt.subplots(figsize=(40, 40), nrows=1)
    cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
    visual = np.sum(feature_vectors, axis=0).reshape((64, 64))
    #visual = feature_vectors[2].reshape((64,64))
    print(np.sum(visual > 0))
    #sns.heatmap(Visual_Samples, annot=True, annot_kws={'size': 5, 'weight': 'bold', 'color': 'blue'},
    #            linewidths=0.05, ax=ax1, cmap=cmap, center=None, robust=False)
    sns.heatmap(visual, linewidths=0.05, ax=ax1, cmap=cmap, center=None, robust=False)
    ax1.set_title('robust=False')
    ax1.set_xlabel('')
    ax1.set_xticklabels([])  # 设置x轴图例为空值
    ax1.set_ylabel('kind')
    ax1.plot()
    plt.show()
    """
    sns.heatmap(visual_InDe, linewidths=0.05, ax=ax2, cmap=cmap, center=None, robust=True)
    ax2.set_title('robust=True')
    ax2.set_xlabel('region')
    ax2.set_ylabel('kind')
    """


if __name__ == "__main__":
    visualization()