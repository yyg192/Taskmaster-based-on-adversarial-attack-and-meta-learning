import os
import sys

import numpy as np
from collections import defaultdict

from abc import ABCMeta, abstractmethod

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from tools.file_operation import *
import re
import collections
from config import COMP
from config import config
from config import USING_DATASET
OPERATOR = {
    # insert
    0: "insert",
    # remove
    1: "remove"
}

INSTR_ALLOWED = {
    OPERATOR[0]: [COMP['Permission'],  #S2
                  COMP['Activity'],
                  COMP['Service'],  #
                  COMP['Receiver'], #
                  COMP['Hardware'],  # S1
                  COMP['Intentfilter'],  #S4
                  COMP['Android_API'],  #S7 S5
                  COMP['User_String']
                  ],
    OPERATOR[1]: [COMP['Activity'],
                  COMP['Service'],
                  COMP['Receiver'],
                  COMP['Provider'],
                  COMP['Android_API'],
                  COMP['User_String']
                  ]
}

MetaInstTemplate = "{Operator}##{Component}##{SpecName}##{Count}"
APIInstrSpecTmpl = "{ClassName}::{ApiName}::{ApiParameter}"
MetaDelimiter = '##'
SpecDelimiter = '::'

section_path = USING_DATASET['malware'] + '_AND_' + USING_DATASET['benware']
normalizer_path = config.get(section_path, 'normalizer')
vocabulary_path = config.get(section_path, 'vocabulary')
vocab_info_path = config.get(section_path, 'vocab_info')

def get_api_ingredient(api_dalvik_code):
    """get class name, method name, parameters from dalvik code line by line"""
    invoke_match = re.search(
        r'(?P<invokeType>invoke\-([^ ]*?)) (?P<invokeParam>([vp0-9,. ]*?)), (?P<invokeObject>L(.*?);|\[L(.*?);)->(?P<invokeMethod>(.*?))\((?P<invokeArgument>(.*?))\)(?P<invokeReturn>(.*?))$',
        api_dalvik_code)

    if invoke_match is None:
        return None, None, None
    else:
        invoke_object = invoke_match.group('invokeObject')
        invoke_method = invoke_match.group('invokeMethod')
        invoke_argument = invoke_match.group('invokeArgument')
        return invoke_object, invoke_method, invoke_argument

class FeatureReverse(object):
    """Abstract base class for inverse feature classes."""
    __metaclass__ = ABCMeta

    def __init__(self, feature_type, feature_mp, use_default_feature = True):
        """
        feature reverse engineering
        :param feature_type: feature type, e.g., drebin
        :param feature_mp: binary bag of words, or counting the occurrence of words
        :param use_default_feature: use the default meta feature information or not, if False the surrogate feature will be leveraged
        """
        self.feature_type = feature_type
        self.feature_mp = feature_mp
        self.use_default_feature = use_default_feature

        self.insertion_array = None
        self.removal_array = None

        self.normalizer = None

    @abstractmethod
    def get_mod_array(self):
        raise NotImplementedError

    @abstractmethod
    def generate_mod_instruction(self, sample_paths, perturbations):
        raise NotImplementedError

def _check_instr(instr):
    elements = instr.strip().split(MetaDelimiter)
    if str.lower(elements[0]) in OPERATOR.values() and \
            str.lower(elements[1]) in COMP.values():
        return True
    else:
        return False


def _check_instructions(instruction_list):
    if not isinstance(instruction_list, list):
        instruction_list = [instruction_list]
    for instr in instruction_list:
        if not _check_instr(instr):
            return False
    return True

def get_word_category(vocabulary, vocabulary_info, defined_comp):
    """
    Get the category for each word in vocabulary, based on the COMP in conf file
    :rtype: object
    """

    def _api_check(dalvik_code_line_list):
        for code_line in dalvik_code_line_list:
            invoke_match = re.search(
                r'(?P<invokeType>invoke\-([^ ]*?)) (?P<invokeParam>([vp0-9,. ]*?)), (?P<invokeObject>L(.*?);|\[L(.*?);)->(?P<invokeMethod>(.*?))\((?P<invokeArgument>(.*?))\)(?P<invokeReturn>(.*?))$',
                code_line)
            if invoke_match is None:
                return defined_comp['Notdefined']
            if invoke_match.group('invokeType') == 'invoke-virtual' or invoke_match.group(
                    'invokeType') == 'invoke-virtual/range' or \
                    invoke_match.group('invokeType') == 'invoke-static' or \
                    invoke_match.group('invokeType') == 'invoke-static/range':
                if invoke_match.group('invokeObject').startswith('Landroid'):
                    return defined_comp['Android_API']
                elif invoke_match.group('invokeObject').startswith('Ljava'):
                    return defined_comp['Java_API']
                else:
                    return defined_comp['Notdefined']
            else:
                return defined_comp['Notdefined']

    word_cat_dict = collections.defaultdict()
    for w in vocabulary:
        if 'ActivityList_' in w:
            word_cat_dict[w] = defined_comp['Activity']
        elif 'RequestedPermissionList_' in w:
            word_cat_dict[w] = defined_comp['Permission']
        elif 'ServiceList_' in w:
            word_cat_dict[w] = defined_comp['Service']
        elif 'ContentProviderList_' in w:
            word_cat_dict[w] = defined_comp['Provider']
        elif 'BroadcastReceiverList_' in w:
            word_cat_dict[w] = defined_comp['Receiver']
        elif 'HardwareComponentsList_' in w:
            word_cat_dict[w] = defined_comp['Hardware']
        elif 'IntentFilterList_' in w:
            word_cat_dict[w] = defined_comp['Intentfilter']
        elif 'UsedPermissionsList_' in w:
            word_cat_dict[w] = defined_comp['Notdefined']
        elif 'RestrictedApiList_' in w:
            word_cat_dict[w] = _api_check(vocabulary_info[w])
        elif 'SuspiciousApiList' in w:
            word_cat_dict[w] = _api_check(vocabulary_info[w])
        elif 'URLDomainList' in w:
            word_cat_dict[w] = defined_comp['User_String']
        else:
            word_cat_dict[w] = defined_comp['Notdefined']
    return word_cat_dict


class DrebinFeatureReverse(FeatureReverse):
    def __init__(self, feature_mp='binary', use_default_feature=True):
        super(DrebinFeatureReverse, self).__init__('drebin',
                                                   feature_mp,
                                                   use_default_feature)
        #load feature infomation
        try:
            self.normalizer = read_pickle(normalizer_path)
            self.vocab = read_pickle(vocabulary_path)
            self.vocab_info = read_pickle(vocab_info_path)
        except Exception as ex:
            raise IOError("Unable to load meta-information of feature.")

    def get_mod_array(self):
        """
        get binary indicator of showing the feature can be either modified or not
        '1' means modifiable and '0' means not
        """
        insertion_array = []
        removal_array = []

        if not os.path.exists(vocabulary_path):
            print("No feature key words at {}.".format(vocabulary_path))
            return insertion_array, removal_array
        if not os.path.exists(vocab_info_path):
            print(
                "No feaure key words description at {}.".format(vocab_info_path))

        word_catagory_dict = get_word_category(self.vocab, self.vocab_info, COMP)
        '''
        这个word_category_dict是一个字典，它的key是特征值，value是下面九种中的一种'Permission','Activity','Service'
        'Receiver','Hardware','Intentfilter','Android_API','User_String','Provider'
        比如:
        {'RequestedPermissionList_android.permission.SEND_SMS': 'permission', 
         'ActivityList_.Main': 'activity',
         'ActivityList_.ReadOffertActivity': 'activity', 
         'ActivityList_.ActivationDoneActivity': 'activity', 
         'IntentFilterList_android.intent.action.MAIN': 'intent-filter', 
         'SuspiciousApiList_Landroid/telephony/SmsManager.sendTextMessage': 'android_api',
         ...
         ...
        }
        这个长度取决于你输入的apks_path中的所有apk总共用了多少特征，我这里取了120个apk，总共涉及了3367个特征
        '''

        insertion_array = np.zeros(len(self.vocab), )
        removal_array = np.zeros(len(self.vocab), )

        for i, word in enumerate(self.vocab):
            cat = word_catagory_dict.get(word)
            if cat is not None:
                if cat in INSTR_ALLOWED[OPERATOR[0]]:  #检查以下vocab，如
                    insertion_array[i] = 1
                else:
                    insertion_array[i] = 0
                if cat in INSTR_ALLOWED[OPERATOR[1]]:
                    removal_array[i] = 1
                else:
                    removal_array[i] = 0
            else:
                raise ValueError("Incompatible value.")

        return insertion_array, removal_array

    def generate_mod_instruction(self, sample_paths, perturbations):
        '''
        generate the instructions for samples in the attack list
        :param sample_paths: the list of file path
        :param perturbations: numerical perturbations on the un-normalized feature space, type: np.ndarray
        :return: {sample_path1: [meta_instruction1, ...], sample_path2: [meta_instruction1, ...],...}
        '''
        assert len(sample_paths) == len(perturbations)

        word_catagory_dict = get_word_category(self.vocab, self.vocab_info, COMP)
        instrs = defaultdict(list)
        for idx, path in enumerate(sample_paths):
            perturb_values = perturbations[idx][perturbations[idx].astype(np.int32) != 0]
            #这个perturb_values其实就是把perturbations的0元素剔除了而已
            perturb_entities = np.array(self.vocab)[perturbations[idx].astype(np.int32) != 0].tolist()
            #哪些需要特征需要perturb一下的，最后这个perturb_entities就是一个数组，元素是字符串，代表需要被perturb的特征的名字列表。
            meta_instrs = []
            for e_idx, e in enumerate(perturb_entities):
                if perturb_values[e_idx] > 0:
                    _operator = OPERATOR[0]  # 'insert'
                else:
                    _operator = OPERATOR[1]  # 'remove'

                # template: "{operator}##{component}##{specName}##{Count}"
                _cat = word_catagory_dict[e]
                _info = list(self.vocab_info[e])[0]
                if _cat is COMP['Notdefined']:
                    continue

                if _cat in [COMP['Android_API'], COMP['Java_API']]:
                    class_name, method_name, params = get_api_ingredient(_info)
                    _info = APIInstrSpecTmpl.format(ClassName=class_name, ApiName=method_name, ApiParameter=params)
                if self.feature_mp == 'count':
                    _count = abs(int(round(perturb_values[e_idx])))
                elif self.feature_mp == 'binary':
                    if _operator == OPERATOR[0]:
                        _count = abs(int(round(perturb_values[e_idx])))
                    else:
                        _count = int(round(perturb_values[e_idx]))
                else:
                    raise ValueError("Allowing feature mapping type: 'count' or 'binary'")

                meta_instr = MetaInstTemplate.format(Operator=_operator, Component=_cat, SpecName=_info, Count=_count)
                if perturb_values[e_idx] > 0:
                    meta_instrs.append(meta_instr)
                else:
                    meta_instrs.insert(0, meta_instr)
            if _check_instructions(meta_instrs):
                instrs[path] = meta_instrs
            else:
                raise AssertionError(" Generate the incorrent intructions.")
        return instrs







