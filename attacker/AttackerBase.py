from tools.feature_reverser import DrebinFeatureReverse
from config import DREBIN_FEATURE_Param
from tools import utils
import numpy as np
class AttackerBase(object):
    def __init__(self):
        self.feature_reverser = DrebinFeatureReverse()
        self.normalizer = self.feature_reverser.normalizer
        self.clip_min, self.clip_max = utils.get_min_max_bound(normalizer=self.normalizer)
        self.scaled_clip_min = utils.normalize_transform(np.reshape(self.clip_min, (1, -1)), normalizer=self.normalizer)
        self.scaled_clip_max = utils.normalize_transform(np.reshape(self.clip_max, (1, -1)), normalizer=self.normalizer)
        self.insertion_perm_array, self.removal_perm_array = self.feature_reverser.get_mod_array()
        self.input_dim = DREBIN_FEATURE_Param['feature_dimension']
        self.output_dim = DREBIN_FEATURE_Param['output_dimension']

