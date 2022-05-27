from tools import utils
import numpy as np
import tensorflow as tf
import tqdm
from tools import utils
import tqdm
from attacker.AttackerBase import AttackerBase

class BlackBox_PGD(AttackerBase):
    def __init__(self,
                 simulator_model,
                 target_model,
                 label_type,
                 ord='l2',
                 maximum_iterations=200,
                 step_size=100.):
        super(BlackBox_PGD, self).__init__()
        self.model = target_model
        self.simulator_model = simulator_model
        self.ForceIteration = tf.placeholder(dtype=tf.bool)
        self.batch_size = 1
        self.ord = ord
        self.label_type = label_type
        self.maximum_iterations = maximum_iterations
        self.force_iteration = False
        self.step_size = step_size
        self.launch_an_attack = self.graph()

    def graph(self):
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
            if self.label_type is "hard_label":
                logit = self.model.get_output_logits(x_adv_tmp)
                logit = tf.one_hot(tf.argmax(logit, axis=1), self.output_dim)
            elif self.label_type is "soft_label":
                logit = self.model.get_output_logits(x_adv_tmp)
            else:
                raise ValueError("The label_type must be soft_label or hard_label")
            loss = tf.losses.sparse_softmax_cross_entropy(logits=logit,
                                                          labels=self.model.y_input)  # average loss, may cause leakage issue
            grad = tf.gradients(loss, x_adv_tmp)[0]
            if self.ord == 'l2':
                perturbations = utils.optimize_linear(grad, tf.to_float(self.step_size), ord=2)
            elif self.ord == 'l-infinity':
                perturbations = utils.optimize_linear(grad, tf.to_float(self.step_size))
            elif self.ord == 'l1':
                raise NotImplementedError("L1 norm based attack is not implemented here.")
            else:
                raise ValueError("'l-infinity' are supported.")

            x_adv_tmp = x_adv_tmp + perturbations
            x_adv_tmp = tf.clip_by_value(x_adv_tmp,
                                         clip_value_min=scaled_min_extended,
                                         clip_value_max=scaled_max_extended)
            x_adv_tmp_discrete = utils.map_to_discrete_domain_TF(self.normalizer, x_adv_tmp)
            predict = self.model.get_output_pred(x_adv_tmp_discrete)

            _attack_success = tf.logical_not(tf.equal(predict, self.model.y_input))[0]

            return i + 1, _attack_success, x_adv_tmp

        iter_num, _attack_success, x_adv_var = tf.while_loop(_cond, _body, (0, False, self.model.x_input),
                                                       maximum_iterations=self.maximum_iterations)
        x_adv_var_discrete = utils.map_to_discrete_domain_TF(self.normalizer, x_adv_var)
        return iter_num, x_adv_var_discrete, _attack_success





    def generate_attack_samples(self,
                                attack_feature_vectors,
                                attack_feature_labels,
                                samples_save_dir,
                                sess):
        input_data = utils.DataProducer(attack_feature_vectors, attack_feature_labels,
                                        batch_size=1, name='test')  #### batch_size 只能为1
        #x_adv_all = []
        attack_success_record = []
        attack_query_record = []

        x_adv_success = []
        x_ori = []
        available_sample_num = 1e-6
        fool_num = 0
        iter_num_sum = 0
        for idx, x_input_var, y_input_var in tqdm.tqdm(input_data.next_batch()):
            predict_x_input = sess.run(self.model.y_pred_output,
                                       feed_dict={self.model.x_input: x_input_var})
            if int(predict_x_input[0]) != int(y_input_var[0]):
                continue

            available_sample_num += 1
            iter_num, x_adv_var, _attack_success = sess.run(self.launch_an_attack,
                                                            feed_dict={self.model.x_input: x_input_var,
                                                                       self.model.y_input: y_input_var,
                                                                       self.ForceIteration: self.force_iteration})
            if bool(_attack_success) is True:
                x_adv_success.append(x_adv_var[0])
                x_ori.append(x_input_var[0])
                fool_num += 1
                iter_num_sum += iter_num
                attack_success_record.append(True)
            else:
                attack_success_record.append(False)
            attack_query_record.append(iter_num)


            #x_adv_all.append(x_adv_var[0])
        #x_adv_all = np.array(x_adv_all)
        x_ori = np.array(x_ori)
        attack_query_record = np.array(attack_query_record)
        attack_success_record = np.array(attack_success_record)
        aux_info = {}
        aux_info["attack_query_recrod"] = attack_query_record
        aux_info["attack_success_record"] = attack_success_record
        x_adv_success = np.array(x_adv_success)
        perturbations_amount_l0 = np.mean(np.sum(np.abs(x_adv_success - x_ori) > 1e-6, axis=1))
        print("average modified features is {:.2f} "
              "and attack_success_rate is {:.2f}"
              "and average_iter_num is {:.2f}".format(perturbations_amount_l0,
                                                      fool_num/available_sample_num*100,
                                                      iter_num_sum/len(x_adv_success)))
        if samples_save_dir is not None:
            utils.dump_pickle(x_adv_success, samples_save_dir+"/x_adv_success")
        return x_adv_success, aux_info


