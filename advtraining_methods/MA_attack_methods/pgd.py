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
    'step_size': 10,
    'ord': 'l2',  # 'l2', 'linfinity',
    'rand_round': False,
    'max_iteration': 1000,
    'force_iteration': False
}


class PGD(AttackerBase):
    def __init__(self,
                 targeted_model,
                 **kwargs
                 ):
        super(PGD, self).__init__()
        self.step_size = DEFAULT_PARAM['step_size']
        self.ord = DEFAULT_PARAM['ord']
        self.rand_round = DEFAULT_PARAM['rand_round']
        self.iterations = DEFAULT_PARAM['max_iteration']
        self.force_iteration = DEFAULT_PARAM['force_iteration']
        self.batch_size = 1
        self.model = targeted_model
        self.parse(**kwargs) ######## 如果不在max中启动的话要把他注释掉
        self.model_inference()

    def parse(self,
              step_size=10,
              ord='l2',
              rand_round=False,
              max_iteration=1000,
              batch_size=50,
              force_iteration=False,
              **kwargs):
        self.step_size = step_size
        self.ord = ord
        self.rand_round = rand_round
        self.iterations = max_iteration
        self.batch_size = 1
        self.force_iteration = force_iteration
        if len(kwargs) > 0:
            warnings.warn("unused hyper parameters.")

    def model_inference(self):
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        self.y_input = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.scaled_max_extended = tf.maximum(
            tf.multiply(self.scaled_clip_max, tf.to_float(self.insertion_perm_array)) +
            tf.multiply(self.scaled_clip_min, 1. - tf.to_float(self.insertion_perm_array)),
            self.x_input
        )
        self.scaled_min_extended = tf.minimum(
            tf.multiply(self.scaled_clip_min, tf.to_float(self.removal_perm_array)) +
            tf.multiply(self.scaled_clip_max, 1. - tf.to_float(self.removal_perm_array)),
            self.x_input
        )

        """
        作为一个还没被攻击就已经分类错误的样本是没有资格作为对抗样本的，所以要在攻击之前先看看这个样本是否分类正确。
        """
        self.output_pred = self.model.get_output_pred(self.x_input)
        """
        攻击迭代一次
        """
        self.x_adv_tmp = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        self.iterate_once_x_adv = self.single_iteration()
        """
        攻击迭代过程中要，每次迭代都要检测一下离散化的对抗样本是否已经成功欺骗了模型
        """
        self.x_adv_tmp_discrete = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        self.predict_discrete_x_adv_tmp = self.model.get_output_pred(self.x_adv_tmp_discrete)

    def single_iteration_old(self, x_adv_tmp_continuous, y_input, scaled_min_extended, scaled_max_extended):
        """ logits """
        output_logits = self.model.get_output_logits(x_adv_tmp_continuous)

        """ loss """
        adv_output_loss = tf.losses.sparse_softmax_cross_entropy(logits=output_logits,
                                                                 labels=y_input)
        """ grad """
        adv_grad = tf.gradients(adv_output_loss, x_adv_tmp_continuous)[0]

        """ perturbations """
        if self.ord == 'l2':
            perturbations = utils.optimize_linear(adv_grad, tf.to_float(self.step_size), ord=2)
        elif self.ord == 'l-infinity':
            perturbations = utils.optimize_linear(adv_grad, tf.to_float(self.step_size))
        elif self.ord == 'l1':
            perturbations = utils.optimize_linear(adv_grad, tf.to_float(self.step_size), ord=1)
        else:
            raise ValueError("'l-infinity' are supported.")

        x_adv_tmp_continuous += perturbations
        x_adv_tmp_continuous = tf.clip_by_value(x_adv_tmp_continuous,
                                                clip_value_min=scaled_min_extended,
                                                clip_value_max=scaled_max_extended)
        return x_adv_tmp_continuous

    def single_iteration(self):
        """ logits """
        output_logits = self.model.get_output_logits(self.x_adv_tmp)

        """ loss """
        adv_output_loss = tf.losses.sparse_softmax_cross_entropy(logits=output_logits,
                                                                 labels=self.y_input)
        """ grad """
        adv_grad = tf.gradients(adv_output_loss, self.x_adv_tmp)[0]

        """ perturbations """
        if self.ord == 'l2':
            perturbations = utils.optimize_linear(adv_grad, tf.to_float(self.step_size), ord=2)
        elif self.ord == 'l-infinity':
            perturbations = utils.optimize_linear(adv_grad, tf.to_float(self.step_size))
        elif self.ord == 'l1':
            perturbations = utils.optimize_linear(adv_grad, tf.to_float(self.step_size), ord=1)
        else:
            raise ValueError("'l-infinity' are supported.")

        x_out = self.x_adv_tmp + perturbations
        x_out = tf.clip_by_value(x_out,
                                 clip_value_min=self.scaled_min_extended,
                                 clip_value_max=self.scaled_max_extended)
        return x_out, perturbations

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
                print("################# model parameter load success ###################")
                sess_close_flag = True

        except IOError as ex:
            raise IOError("PGD attack: Failed to load data and model parameters.")

        fool_num = 0
        iter_num_sum = 0
        x_adv = []
        x_ori = []
        x_adv_ori = []
        available_sample_num = 1e-6
        attack_success_record = np.zeros((attack_feature_vectors.shape[0]), dtype=bool)
        with sess.as_default():
            for idx, x_input_var, y_input_var in tqdm.tqdm(input_data.next_batch()):
                predict_clean_x_before_attack = sess.run(self.output_pred, feed_dict={self.x_input: x_input_var})
                if int(predict_clean_x_before_attack[0] != int(y_input_var[0])):
                    x_adv_ori.append(x_input_var[0])
                    continue
                available_sample_num += 1
                x_adv_tmp_var = x_input_var
                for iter_num in range(0, self.iterations, 1):
                    x_adv_tmp_var, perturbations = sess.run(self.iterate_once_x_adv,
                                             feed_dict={self.x_input: x_input_var,
                                                        self.x_adv_tmp: x_adv_tmp_var,
                                                        self.y_input: y_input_var,
                                                        })
                    x_adv_tmp_discrete_var = utils.map_to_discrete_domain(self.normalizer, x_adv_tmp_var)
                    predict_discrete_x_adv_tmp = sess.run(self.predict_discrete_x_adv_tmp,
                                                          feed_dict={self.x_adv_tmp_discrete: x_adv_tmp_discrete_var})
                    if int(predict_discrete_x_adv_tmp[0]) != int(y_input_var[0]):
                        fool_num += 1
                        x_ori.append(x_input_var[0])
                        x_adv.append(x_adv_tmp_discrete_var[0])
                        iter_num_sum += iter_num
                        iter_num_sum += 1
                        attack_success_record[idx] = True
                        x_adv_ori.append(x_adv_tmp_discrete_var[0])
                        break
                    if int(iter_num) == self.iterations-1:
                        x_adv_ori.append(x_input_var[0])


        if sess_close_flag:
            sess.close()
        x_adv = np.array(x_adv)
        x_ori = np.array(x_ori)
        x_adv_ori = np.array(x_adv_ori)
        utils.dump_pickle(x_adv, "E:/jupyter_official/x_adv_pgd_l2")
        utils.dump_pickle(x_ori, "E:/jupyter_official/x_ori_pgd_l2")
        adv_accuracy = 1 - fool_num / available_sample_num
        print("adv_accuracy: {:.2f}% ".format(adv_accuracy * 100))
        print("average iter_num: {:.2f}".format(iter_num_sum / available_sample_num))
        if x_adv.shape[0] != 0:
            perturbations_amount_l0 = np.mean(np.sum(np.abs(x_adv - x_ori) > 1e-6, axis=1))
            print("The number of modified features is {:.2f}".format(perturbations_amount_l0))
        else:
            print("No sample is successfully attacked")
        return x_adv, x_ori, x_adv_ori, attack_success_record

