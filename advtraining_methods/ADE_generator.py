import os
import sys
import warnings

import tensorflow as tf
import numpy as np
from collections import defaultdict
from config import model_Param_dict
from tools.file_operation import read_pickle
from sklearn.model_selection import train_test_split
from config import config
from attacker.AttackerBase import AttackerBase
from keras.utils import to_categorical
import tqdm
from models.teacher_model import teacher_model
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_root)
from advtraining_methods.MA_attack_methods.pgd import PGD
from advtraining_methods.MA_attack_methods.pgdl1 import PGDl1
from advtraining_methods.MA_attack_methods.pgd_adam import PGDAdam

DEFAULT_PARAM = {
    'random_seed': 0,
    'iteration': 5,
    'call_sp': True,
    'use_fast_version': False,
    'varepsilon': 1e-9,
    'attack_names': ['pgdl1', 'pgdl2', 'pgdlinf', 'pgd_adam']
}
max_param_dict = {'random_seed': 0,
                  'iteration': 5,
                  'call_saltandpepper': False,
                  'use_fast_version': False,
                  'varepsilon': 1e-9,
                  'attack_names': ['pgdl1', 'pgdl2', 'pgdlinf', 'pgd_adam']
                  }

attack_params_dict = {
    'pgdl1': {'k': 1, 'step_size': 1., 'max_iteration': 1000, 'batch_size': 1,
              'force_iteration': False},
    'pgdl2': {'step_size': 10., 'ord': 'l2', 'max_iteration': 1000, 'batch_size': 1,
              'force_iteartion': False},
    'pgdlinf': {'step_size': 0.01, 'ord': 'l-infinity', 'max_iteration': 1000,
                'batch_size': 1, 'force_iteration': False},
    'pgd_adam': {'learning_rate': 0.01, 'max_iteration': 1000, 'batch_size': 50,
                 'force_iteration': False}
}
attack_methods_dict = defaultdict()

def _main():
    malware_dataset_name = "virustotal_2018_5M_17M"
    benware_dataset_name = "androzoo_benware_3M_17M"
    advtraining_method = "pgdl2"
    exp_name = "4096_" + malware_dataset_name + "_AND_" + benware_dataset_name
    adv_train_root = config.get("advtraining.drebin", "advtraining_drebin_root") + "/" + advtraining_method + \
                     "_" + exp_name
    ori_malware_features_vectors = read_pickle(config.get(malware_dataset_name, "sample_vectors"))
    ori_malware_features_labels = to_categorical(np.ones(ori_malware_features_vectors), num_classes=2)
    target_model_load_dir = adv_train_root + "/target_model/adv0"
    target_graph = tf.Graph()
    target_sess = tf.Session(graph=target_graph)
    with target_graph.as_default():
        with target_sess.as_default():
            target_model = teacher_model(hyper_parameter=model_Param_dict['target_model'],
                                         model_name='target_model',
                                         is_trainable=True)
            MAX_attack = MAX(targeted_model=target_model,
                             **max_param_dict
                             )
            target_saver = tf.train.Saver()
            target_sess.run(tf.global_variables_initializer())
            target_model.load_param(target_model_load_dir, target_sess, target_saver)


class MAX(AttackerBase):
    def __init__(self,
                 targeted_model,
                 **kwargs):
        super(MAX, self).__init__()
        self.model = targeted_model
        self.iteration = DEFAULT_PARAM['iteration']
        self.attack_names = DEFAULT_PARAM['attack_names']
        self.call_sp = DEFAULT_PARAM['call_sp']
        self.varepsilon = DEFAULT_PARAM['varepsilon']
        self.attack_seq_selected = defaultdict(list)
        self.attack_mthds_dict = attack_methods_dict
        self.parse(**kwargs)
        self.model_inference()
        # attacks
        self.attack_mthds_dict['pgdl1'] = \
            PGDl1(targeted_model,
                  **attack_params_dict['pgdl1']
                  )
        self.attack_mthds_dict['pgdl2'] = \
            PGD(targeted_model,
                **attack_params_dict['pgdl2']
                )
        self.attack_mthds_dict['pgdlinf'] = \
            PGD(targeted_model,
                **attack_params_dict['pgdlinf']
                )
        self.attack_mthds_dict['pgd_adam'] = \
            PGDAdam(targeted_model,
                    **attack_params_dict['pgd_adam']
                    )

    def parse(self, random_seed=0, iteration=1, call_saltandpepper=True,
              use_fast_version=False, varepsilon=1e-9, attack_names=None, **kwargs):
        self.random_seed = random_seed
        self.iteration = iteration
        self.call_sp = call_saltandpepper
        self.use_fast_version = use_fast_version
        self.varepsilon = varepsilon
        self.attack_names = attack_names

        if len(kwargs) > 0:
            warnings.warn("unused hyper parameters.")

    def model_inference(self):
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        self.y_input = tf.placeholder(dtype=tf.int32, shape=[None, ])

    def get_start_point(self, attack_feature_vectors, attack_feature_labels, sess=None):
        x_perturb = attack_feature_vectors.copy()
        min_prob = np.ones(attack_feature_labels.shape)  # shape: (2733, )
        for iter_idx in range(self.iteration):
            adv_x_list = []
            spec_names = []
            attack_success_record = []
            for a in self.attack_names:
                if a in self.attack_mthds_dict.keys():
                    _1, _2, adv_x, attack_success_record_each = self.attack_mthds_dict[a].perturb(x_perturb,
                                                                                                  self.attack_feature_labels,
                                                                                                  sess)
                    adv_x_list.append(adv_x)
                    spec_names.append(a)
                    attack_success_record.append(attack_success_record_each)
                else:
                    warnings.warn("No attack:{}".format(a))

            attack_num = len(adv_x_list)  # 用多少种攻击方法攻击了self.attack_feature_vectors
            all_adv_x = np.concatenate(
                adv_x_list)  # (attack_num, sample_num, input_dim)的变为一个(attack_num*sample_num,input_dim)的
            """在此处修改"""
            output_logits = self.model.get_output_logits(self.x_input)
            combine_logits = output_logits  # shape(?, 2)
            y_proba = tf.nn.softmax(combine_logits)  # shape(?, 2)
            _y_prob = sess.run(y_proba,
                               feed_dict={self.x_input: all_adv_x})  #
            # _y_prob的shape应该为(attack_num*sample_num, 2)
            all_adv_trans = np.transpose(np.reshape(all_adv_x, (attack_num,
                                                                self.attack_feature_vectors.shape[0],
                                                                self.attack_feature_vectors.shape[1])),
                                         (1, 0, 2))
            # all_adv_x是(sample_num*attack_num, input_dim)
            # adv_x_list是(attack_num, sample_num, input_dim)
            # all_adv_trans就是(sample_num, attack_num, input_dim)
            _y_prob_trans = np.transpose(np.reshape(_y_prob, (attack_num,
                                                              self.attack_feature_vectors.shape[0],
                                                              _y_prob.shape[1])),
                                         (1, 0, 2))
            # _y_prob是(attack_num*sample_num, 2) _y_prob_trans是(sample_num, attack_num, 2)

            for i in range(len(self.attack_feature_vectors)):
                gt = int(self.attack_feature_labels[i])
                min_v = np.min(_y_prob_trans[i, :, gt])  # 欺骗性越强的样本它的_y_prob_trans[i, attack_method, 1]越小
                # 这里的目的是为了筛选出每个样本的N个方法中能产生欺骗性最强的样本对应恶意类别的logits
                if min_v < min_prob[i]:  # 第一轮iteration当然min_v 会小于min_prob[i]了
                    min_prob[i] = min_v
                    min_ind = int(np.argmin(_y_prob_trans[i, :, gt]))
                    x_perturb[i, :] = all_adv_trans[i, min_ind, :]
                    self.attack_seq_selected[i].append(spec_names[min_ind])  # self.attack_seq_selected[i]是个default_dict
                else:
                    self.attack_seq_selected[i].append(' ')
                    continue

        return x_perturb

    def perturb(self, sess=None):
        try:
            # load model parameters
            sess_close_flag = False
            if sess is None:
                cur_checkpoint = tf.train.latest_checkpoint(self.model.save_dir)
                config_gpu = tf.ConfigProto(log_device_placement=True)
                config_gpu.gpu_options.allow_growth = True
                sess = tf.Session(config=config_gpu)
                saver = tf.train.Saver()
                saver.restore(sess, cur_checkpoint)
                sess_close_flag = True
        except IOError as ex:
            raise IOError("Failed to load data and model parameters.")

        with sess.as_default():
            x_adv_init = self.get_start_point(dataX, ground_truth_labels, sess)

            assert x_adv_init.shape == dataX.shape
            x_adv = np.copy(x_adv_init)

            if self.call_sp:
                for idx in range(len(x_adv_init)):
                    feat_vector = dataX[idx: idx + 1]
                    adv_feat_vector = x_adv_init[idx: idx + 1]

                    shape = feat_vector.shape
                    N = feat_vector.size

                    orig_feats = feat_vector.reshape(-1)
                    adv_feats = adv_feat_vector.reshape(-1)

                    np.random.seed(self.random_seed)
                    while True:
                        # draw random shuffling of all indices
                        indices = list(range(N))
                        np.random.shuffle(indices)

                        for index in indices:
                            # start trial
                            old_value = adv_feats[index]
                            new_value = orig_feats[index]
                            if old_value == new_value:
                                continue
                            adv_feats[index] = new_value

                            # check if still adversarial
                            _acc = sess.run(self.model.accuracy, feed_dict={
                                self.model.x_input: adv_feats.reshape(shape),
                                self.model.y_input: ground_truth_labels[idx: idx + 1],
                                self.model.is_training: False
                            })

                            if _acc <= 0.:
                                x_adv[idx: idx + 1] = adv_feats.reshape(shape)
                                break

                            # if not, undo change
                            adv_feats[index] = old_value
                        else:
                            print("No features can be flipped by adversary successfully")
                            break

            if self.normalizer is not None:
                x_adv = np.rint(normalize_inverse(x_adv, self.normalizer))
                # check again
                x_adv_normalized = normalize_transform(x_adv, self.normalizer)
            else:
                x_adv_normalized = np.rint(x_adv)
            if self.verbose:
                accuracy = utils.test_func(sess, self.model, x_adv_normalized, ground_truth_labels, batch_size=50)
                print("The classification accuracy is {:.5} on adversarial feature vectors.".format(accuracy))

                perturbations_amount_l0 = np.mean(np.sum(np.abs(x_adv_normalized - dataX) > 1e-6, axis=1))
                perturbations_amount_l1 = np.mean(np.sum(np.abs(x_adv_normalized - dataX), axis=1))
                perturbations_amount_l2 = np.mean(np.sqrt(np.sum(np.square(x_adv_normalized - dataX), axis=1)))
                print("\t The average l0 norm of perturbations is {:5}".format(perturbations_amount_l0))
                print("\t The average l1 norm of perturbations is {:5}".format(perturbations_amount_l1))
                print("\t The average l2 norm of perturbations is {:5}".format(perturbations_amount_l2))

            if sess_close_flag:
                sess.close()
        utils.dump_json(self.attack_seq_selected, './metainfo.json')
        # dump to disk
        return dataX, x_adv_normalized, ground_truth_labels


if __name__ == "__main__":
    _main()
