from tools import utils
import numpy as np
from attacker.AttackerBase import AttackerBase
import tensorflow as tf
import tqdm
import random


class pgd_linfinity_generator(AttackerBase):
    def __init__(self,
                 target_model,
                 maximum_iterations,
                 force_iteration,
                 use_search_domain,
                 random_mask,
                 mask_rate,
                 step_size=0.2):
        super(pgd_linfinity_generator, self).__init__()
        self.model = target_model
        self.scaled_clip_min_TS = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        self.scaled_clip_max_TS = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        self.ForceIteration = tf.placeholder(dtype=tf.bool)
        self.random_mask_TS = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        self.batch_size = 1
        self.ord = "l-infinity"
        self.random_mask = random_mask
        self.mask_rate = mask_rate  # 0.3
        self.maximum_iterations = maximum_iterations
        self.use_search_domain = use_search_domain
        self.force_iteration = force_iteration
        self.step_size = step_size

        self.launch_an_attack_teacher = self.graph_teacher()
        self.launch_an_attack_substitute = self.graph_substitute()

    def graph_teacher(self):
        scaled_max_extended = tf.maximum(
            tf.multiply(self.scaled_clip_max, tf.to_float(self.insertion_perm_array)) +
            tf.multiply(self.scaled_clip_min, 1. - tf.to_float(self.insertion_perm_array)),
            self.model.x_input
        )
        scaled_min_extended = tf.minimum(
            tf.multiply(self.scaled_clip_min, tf.to_float(self.removal_perm_array)) +
            tf.multiply(self.scaled_clip_max, 1. - tf.to_float(self.removal_perm_array)),
            self.model.x_input
        )

        def _cond(i, _attack_success, useless):
            return tf.logical_and(
                tf.logical_or(self.ForceIteration, tf.logical_not(_attack_success)),
                tf.less(i, self.maximum_iterations)
            )

        def _body(i, _attack_success, x_adv_tmp):
            output_logits = self.model.get_output_logits(x_adv_tmp)

            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=output_logits,
                                                        labels=self.model.y_input)
                # average loss, may cause leakage issue
            )

            grad = tf.gradients(loss, x_adv_tmp)[0]
            perturbations = utils.optimize_linear(grad, tf.to_float(self.step_size))
            x_adv_tmp = x_adv_tmp + perturbations
            x_adv_tmp = tf.clip_by_value(x_adv_tmp,
                                         clip_value_min=scaled_min_extended,
                                         clip_value_max=scaled_max_extended)
            x_adv_tmp_discrete = utils.map_to_discrete_domain_TF(self.normalizer, x_adv_tmp)
            predict = self.model.get_output_pred(x_adv_tmp_discrete)
            _attack_success = \
            tf.logical_not(tf.equal(predict, tf.argmax(self.model.y_input, axis=-1, output_type=tf.int32)))[0]

            return i + 1, _attack_success, x_adv_tmp

        iter_num, attack_success, x_adv_var = tf.while_loop(_cond, _body, (0, False, self.model.x_input),
                                                            maximum_iterations=self.maximum_iterations)
        x_adv_discrete = utils.map_to_discrete_domain_TF(self.normalizer, x_adv_var)
        return iter_num, x_adv_var, attack_success

    def graph_substitute(self):
        def _cond(i, _attack_success, useless):
            return tf.logical_and(
                tf.logical_or(self.ForceIteration, tf.logical_not(_attack_success)),
                tf.less(i, self.maximum_iterations)
            )

        def _body(i, _attack_success, x_adv_tmp):
            output_logits = self.model.get_output_logits(x_adv_tmp)

            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=output_logits,
                                                        labels=self.model.y_input)
                # average loss, may cause leakage issue
            )
            grad = tf.gradients(loss, x_adv_tmp)[0]
            if self.ord == 'l2':
                perturbations = utils.optimize_linear(grad, tf.to_float(self.step_size), ord=2)
            elif self.ord == 'l-infinity':
                perturbations = utils.optimize_linear(grad, tf.to_float(self.step_size))
            elif self.ord == 'l1':
                raise NotImplementedError("L1 norm based attack is not implemented here.")
            else:
                raise ValueError("'l-infinity' are supported.")

            perturbations = tf.multiply(perturbations, self.random_mask_TS)
            x_adv_tmp = x_adv_tmp + perturbations
            x_adv_tmp = tf.clip_by_value(x_adv_tmp,
                                         clip_value_min=self.scaled_clip_min_TS,
                                         clip_value_max=self.scaled_clip_max_TS)
            predict = self.model.get_output_pred(x_adv_tmp)
            _attack_success = \
            tf.logical_not(tf.equal(predict, tf.argmax(self.model.y_input, axis=-1, output_type=tf.int32)))[0]

            return i + 1, _attack_success, x_adv_tmp

        iter_num, attack_success, x_adv_var = tf.while_loop(_cond, _body, (0, False, self.model.x_input),
                                                            maximum_iterations=self.maximum_iterations)
        return iter_num, x_adv_var, attack_success

    def generate_attack_samples_teacher(self,
                                        attack_feature_vectors,
                                        attack_feature_labels,
                                        samples_save_dir,
                                        sess,
                                        flag="malware"):
        """
        teacher专用
        """
        input_data = utils.DataProducer(attack_feature_vectors, attack_feature_labels,
                                        batch_size=1, name='test')  #### batch_size 只能为1
        x_adv_success = []
        x_ori = []
        x_adv_all = []
        available_sample_num = 1e-6
        fool_num = 1e-6
        iter_num_sum = 0
        for idx, x_input_var, y_input_var in tqdm.tqdm(input_data.next_batch()):
            predict_x_input = sess.run(self.model.y_pred_output,
                                       feed_dict={self.model.x_input: x_input_var})
            if int(predict_x_input[0]) != int(np.argmax(y_input_var, axis=-1)[0]):
                x_adv_all.append(x_input_var[0])
                continue

            available_sample_num += 1
            iter_num, x_adv_var, _attack_success = sess.run(self.launch_an_attack_teacher,
                                                            feed_dict={self.model.x_input: x_input_var,
                                                                       self.model.y_input: y_input_var,
                                                                       self.ForceIteration: self.force_iteration})
            x_adv_all.append(x_adv_var[0])
            if bool(_attack_success) is True:
                x_adv_success.append(x_adv_var[0])
                x_ori.append(x_input_var[0])
                fool_num += 1
                iter_num_sum += iter_num

        x_ori = np.array(x_ori)
        x_adv_success = np.array(x_adv_success)
        x_adv_all = np.array(x_adv_all)
        perturbations_amount_l0 = np.mean(np.sum(np.abs(x_adv_success - x_ori) > 1e-6, axis=1))
        print("average modified samples number is :{:.2f} "
              "attack_success_rate is {:.2f}".format(perturbations_amount_l0,
                                                     fool_num / available_sample_num * 100))
        if samples_save_dir is not None:
            utils.dump_pickle(x_adv_success, samples_save_dir + "/x_adv_success_"+flag)
            utils.dump_pickle(x_adv_all, samples_save_dir + "/x_adv_all_"+flag)
            utils.dump_pickle(x_ori, samples_save_dir + "/x_ori_"+flag)
        return x_adv_success, x_adv_all

    def generate_attack_samples_substitute(self,
                                           attack_feature_vectors,
                                           attack_feature_labels,
                                           require_sample_nums,
                                           sess):
        """
            simulator专用！！
            """
        if len(attack_feature_vectors.shape) == 1:
            attack_feature_vectors = np.array([attack_feature_vectors])
            attack_feature_labels = np.array([attack_feature_labels])
        elif len(attack_feature_vectors.shape) == 2:
            assert (len(attack_feature_vectors) == 1)
        else:
            raise ValueError("Only one sample can be received")
        input_data = utils.DataProducer(attack_feature_vectors, attack_feature_labels,
                                        batch_size=1, name='test')  #### batch_size 只能为1
        x_adv_all = []
        x_ori = []
        logits = []
        available_sample_num = 1e-6
        iter_num_sum = 0
        fool_num = 1e-6
        for idx, x_input_var, y_input_var in input_data.next_batch():
            if self.use_search_domain is True:
                scaled_max_extended = np.maximum(
                    np.multiply(self.scaled_clip_max,
                                self.insertion_perm_array) +  # upper bound for positions allowing perturbations
                    np.multiply(self.scaled_clip_min, 1. - self.insertion_perm_array),
                    # may be useful to reset the lower bound
                    x_input_var  # upper bound for positions no perturbations allowed
                )
                scaled_min_extended = np.minimum(
                    np.multiply(self.scaled_clip_min, self.removal_perm_array) +
                    np.multiply(self.scaled_clip_max, 1. - self.removal_perm_array),
                    x_input_var
                )
            else:
                scaled_max_extended = self.scaled_clip_max
                scaled_min_extended = self.scaled_clip_min
            for _ in range(require_sample_nums):
                available_sample_num += 1
                if self.random_mask is True:
                    random_mask = random.sample(range(0, self.input_dim), int(self.input_dim * self.mask_rate))
                    mask_arr = np.ones((1, self.input_dim))
                    mask_arr[0][random_mask] = 0
                else:
                    mask_arr = np.ones((1, self.input_dim))
                iter_num, x_adv_var, _attack_success = sess.run(self.launch_an_attack_substitute,
                                                                feed_dict={self.model.x_input: x_input_var,
                                                                           self.model.y_input: y_input_var,
                                                                           self.scaled_clip_min_TS: scaled_min_extended,
                                                                           self.scaled_clip_max_TS: scaled_max_extended,
                                                                           self.ForceIteration: self.force_iteration,
                                                                           self.random_mask_TS: mask_arr})
                if bool(_attack_success) is True:
                    iter_num_sum += iter_num
                    fool_num += 1

                logit = sess.run(self.model.softmax_output_logits, feed_dict={self.model.x_input: x_adv_var})[0]
                x_adv_all.append(x_adv_var[0])
                x_ori.append(x_input_var[0])
                logits.append(logit)

        x_adv_all = np.array(x_adv_all)
        x_ori = np.array(x_ori)
        logits = np.array(logits)
        logits_diff = np.mean(np.abs(logits[:, 0] - logits[:, 1]))
        perturbations_amount_l0 = np.mean(np.sum(np.abs(x_adv_all - x_ori) > 1e-6, axis=1))
        perturbations_amount_l1 = np.mean(np.sum(np.abs(x_adv_all - x_ori), axis=1))
        """
        print(
            "perturb_l0 is {:.2f} and perturb_l1 is {:.2f} and average_adv_logits diff is {:.2f} and attack_success_rate "
            ": {:.2f}".format( 
                perturbations_amount_l0,
                perturbations_amount_l1,
                logits_diff,
                fool_num / available_sample_num * 100))
        """
        visualization_info = {}
        visualization_info['logits_diff'] = logits_diff
        visualization_info['perturbations_amount_l0'] = perturbations_amount_l0
        visualization_info['perturbations_amount_l1'] = perturbations_amount_l1
        visualization_info['attack_success_rate'] = fool_num / available_sample_num * 100
        visualization_info['average_iter_num'] = iter_num_sum / fool_num
        return x_adv_all, visualization_info
