"""
Projected gradient descent, l2, l-infinity norm based attack
link: https://adversarial-ml-tutorial.org
malware related paper: https://arxiv.org/abs/2004.07919
"""

import os
import sys
import warnings
import tqdm
import tensorflow as tf
import numpy as np
from tools import utils
from attacker.AttackerBase import AttackerBase
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_root)

DEFAULT_PARAM = {
    'k': 1,  # we'd better set k = 1 if input is binary representations
    'step_size': 1,
    'max_iteration': 1000,
    'batch_size': 1,
    'force_iteration': False  # do not terminate the iteration even the miss-classification happens
}


class PGDl1(AttackerBase):
    def __init__(self,
                 targeted_model,
                 **kwargs):
        super(PGDl1, self).__init__()
        self.model = targeted_model
        self.k = DEFAULT_PARAM['k']
        self.step_size = DEFAULT_PARAM['step_size']
        self.batch_size = 1
        self.iterations = DEFAULT_PARAM['max_iteration']
        self.force_iteration = DEFAULT_PARAM['force_iteration']
        self.parse(**kwargs)
        self.model_inference()
        self.launch_an_attack = self.graph()

    def parse(self, k=1,
              step_size=1.,
              max_iteration=50,
              batch_size=128,
              force_iteration=False,
              **kwargs):
        self.k = k
        self.step_size = step_size
        self.iterations = max_iteration
        self.batch_size = 1
        self.force_iteration = force_iteration

        if len(kwargs) > 0:
            warnings.warn("unused hyper parameters.")

    def model_inference(self):
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        self.y_input = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.output_pred = self.model.get_output_pred(self.x_input)
        self.ForceIteration = tf.placeholder(dtype=tf.bool)

    def graph(self):
        # boundary
        scaled_max_extended = tf.maximum(
            tf.multiply(self.scaled_clip_max, tf.to_float(self.insertion_perm_array)) +
            tf.multiply(self.scaled_clip_min, 1. - tf.to_float(self.insertion_perm_array)),
            self.x_input
        )
        scaled_min_extended = tf.minimum(
            tf.multiply(self.scaled_clip_min, tf.to_float(self.removal_perm_array)) +
            tf.multiply(self.scaled_clip_max, 1. - tf.to_float(self.removal_perm_array)),
            self.x_input
        )

        increase_domain = tf.reshape(self.x_input < scaled_max_extended,
                                     [-1, self.input_dim])
        decrease_domian = tf.reshape(self.x_input > scaled_min_extended,
                                     [-1, self.input_dim])

        search_domain = tf.cast(tf.logical_or(increase_domain, decrease_domian), tf.float32)

        def _cond(i, _domain_out, _exist_modifiable_feature, _attack_success, useless):
            return tf.logical_and(
                tf.logical_or(self.ForceIteration, tf.logical_not(_attack_success)),
                tf.logical_and(tf.less(i, self.iterations), _exist_modifiable_feature)
            )

        def single_iteration(i, domain_in, useless_1, useless_2, x_adv_tmp):
            output_logits = self.model.get_output_logits(x_adv_tmp)

            loss = tf.losses.sparse_softmax_cross_entropy(logits=output_logits,
                                                          labels=self.y_input)

            grad = tf.gradients(loss, x_adv_tmp)[0]
            abs_grad = tf.reshape(tf.abs(grad), (-1, self.input_dim))
            threshold = 0.

            tmp_increase_domain = tf.reshape(tf.less(x_adv_tmp, scaled_max_extended), (-1, self.input_dim))
            tmp_increase_domain = tf.logical_and(tf.cast(domain_in, tf.bool), tmp_increase_domain)
            tmp_domain1 = tf.logical_and(tf.greater(grad, tf.to_float(threshold)),
                                         tmp_increase_domain)

            tmp_decrease_domain = tf.reshape(tf.greater(x_adv_tmp, scaled_min_extended), (-1, self.input_dim))
            tmp_decrease_domain = tf.logical_and(tf.cast(domain_in, tf.bool), tmp_decrease_domain)
            tmp_domain2 = tf.logical_and(tf.less(grad, tf.to_float(-1 * threshold)),
                                         tmp_decrease_domain)

            tmp_search_domain = tf.cast(tf.logical_or(tmp_domain1, tmp_domain2), tf.float32)
            score_mask = tf.cast(abs_grad > 0., tf.float32) * tmp_search_domain

            abs_grad_mask = abs_grad * score_mask
            top_k_v, top_k_idx = tf.nn.top_k(abs_grad_mask, k=self.k)
            changed_pos = tf.reduce_sum(tf.one_hot(top_k_idx, depth=self.input_dim), axis=1)
            perturbations = tf.sign(grad) * changed_pos * tmp_search_domain
            # positions corresponds to the changed value will be neglected
            domain_out = domain_in - changed_pos

            exist_modifiable_feature = (tf.reduce_sum(domain_in, axis=1) >= 1) #(?, )
            #注意这里是以domain_in判断，而不是domain_out
            exist_modifiable_feature_float = tf.reshape(tf.cast(exist_modifiable_feature, tf.float32),
                                                        shape=[-1, 1]) #(?, 1)
            to_mod = perturbations * exist_modifiable_feature_float
            to_mod_reshape = tf.reshape(
                to_mod, shape=([-1] + x_adv_tmp.shape[1:].as_list()))

            x_out = x_adv_tmp + to_mod_reshape * self.step_size / tf.maximum(
                tf.reduce_sum(to_mod_reshape, -1, keepdims=True), self.k)

            x_out = tf.clip_by_value(x_out,
                                     clip_value_min=scaled_min_extended,
                                     clip_value_max=scaled_max_extended)
            x_adv_tmp_discrete = utils.map_to_discrete_domain_TF(self.normalizer, x_out)
            predict_x_adv_tmp_discrete = self.model.get_output_pred(x_adv_tmp_discrete)
            _attack_success = tf.logical_not(tf.equal(predict_x_adv_tmp_discrete, self.y_input))[0]
            #attack_success:shape() exist_modifiable_feature:shape()
            return i + 1, domain_out, exist_modifiable_feature[0], _attack_success, x_out
        """ return i + 1, domain_out, exist_modifiable_feature, _attack_success, x_out """

        iter_num, _2, exist_modifiable_feature, attack_success, x_adv = \
            tf.while_loop(_cond,
                          single_iteration,
                          (0, search_domain, True, False, self.x_input),
                          maximum_iterations=self.iterations,
                          back_prop=False)

        return iter_num, exist_modifiable_feature, attack_success, x_adv

    def perturb(self, attack_feature_vectors, attack_feature_labels, sess=None):
        # TF tensor
        try:
            input_data = utils.DataProducer(attack_feature_vectors, attack_feature_labels,
                                            batch_size=self.batch_size, name='test')

            # load model parameters
            sess_close_flag = False

            if sess is None:
                cur_checkpoint = tf.train.latest_checkpoint(self.model.final_checkpoint_dir)
                config_gpu = tf.ConfigProto(log_device_placement=False)
                config_gpu.gpu_options.allow_growth = True
                sess = tf.Session(config=config_gpu)
                saver = tf.train.Saver()
                saver.restore(sess, cur_checkpoint)
                sess_close_flag = True
        except Exception:
            raise IOError("l1 norm PGD attack: Failed to load data and model parameters.")
        attack_success_record = np.zeros((attack_feature_vectors.shape[0]), dtype=bool)
        x_adv = []
        x_ori = []
        x_adv_ori = []
        fool_num = 0
        iter_num_sum = 0
        exist_num = 0
        available_sample_num = 1e-6
        with sess.as_default():
            for idx, x_input_var, y_input_var in tqdm.tqdm(input_data.next_batch()):
                predict_clean_x_before_attack = sess.run(self.output_pred,
                                                         feed_dict={self.x_input: x_input_var})
                if int(predict_clean_x_before_attack[0]) != int(y_input_var[0]):
                    x_adv_ori.append(x_input_var[0])
                    continue

                available_sample_num += 1
                iter_num, exist_modifiable_feature, attack_success, x_adv_var = \
                    sess.run(self.launch_an_attack,
                             feed_dict={self.x_input: x_input_var,
                                        self.y_input: y_input_var,
                                        self.ForceIteration: self.force_iteration})
                if bool(exist_modifiable_feature) is False:
                    exist_num += 1
                    continue
                if bool(attack_success) is True:
                    x_adv_discrete_var = utils.map_to_discrete_domain(self.normalizer, x_adv_var)
                    x_adv.append(x_adv_discrete_var[0])
                    x_ori.append(x_input_var[0])
                    fool_num += 1
                    iter_num_sum += iter_num
                    attack_success_record[idx] = True
                    x_adv_ori.append(x_adv_discrete_var[0])
                else:
                    x_adv_ori.append(x_input_var[0])
            if sess_close_flag:
                sess.close()

        x_adv = np.array(x_adv)
        x_ori = np.array(x_ori)
        utils.dump_pickle(x_adv, "E:/jupyter_official/pgdl1_xadv")
        utils.dump_pickle(x_ori, "E:/jupyter_official/pgdl1_xori")
        x_adv_ori = np.array(x_adv_ori)
        adv_accuracy = 1 - fool_num / available_sample_num
        print("adv_accuracy: {:.2f}% ".format(adv_accuracy * 100))
        print("average iter_num: {:.2f}".format(iter_num_sum / available_sample_num))
        print("exist_num: {}".format(exist_num))
        if x_adv.shape[0] != 0:
            perturbations_amount_l0 = np.mean(np.sum(np.abs(x_adv - x_ori) > 1e-6, axis=1))
            print("The number of modified features is {:.2f}".format(perturbations_amount_l0))
        else:
            print("No sample is successfully attacked")

        return x_adv, x_ori, x_adv_ori, attack_success_record
