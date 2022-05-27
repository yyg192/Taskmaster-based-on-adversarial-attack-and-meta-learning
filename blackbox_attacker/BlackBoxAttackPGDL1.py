import tensorflow as tf
from attacker.AttackerBase import AttackerBase
from tools import utils
from keras.utils import to_categorical
from config import modelTarget_Param
from models.teacher_model import teacher_model
from models.simulator_model import SimulatorModel
from config import config
from config import model_Param_dict
from config import DREBIN_FEATURE_Param
from advtraining_methods.pgdl2_generator import pgdl2_generator
from advtraining_methods.pgdl1_generator import pgdl1_generator
from advtraining_methods.pgd_linfinity_generator import pgd_linfinity_generator
from tools.DataProducer import DataProducer
from tools.utils import read_pickle
from sklearn.model_selection import train_test_split
import tqdm
import numpy as np


class BlackBoxAttackPGDL1(AttackerBase):
    def __init__(self,
                 target_model,
                 substitute_model,
                 substitute_model_load_dir,
                 label_type,
                 step_size,
                 warmup_iterations=32,
                 maximum_query_iterations=200,
                 inner_adv_iterations=20,
                 substitute_training_interval=1,
                 ):
        super(BlackBoxAttackPGDL1, self).__init__()
        self.target_model = target_model
        self.substitute_model = substitute_model
        self.warmup_iterations = warmup_iterations
        self.maximum_query_iterations = maximum_query_iterations
        self.inner_adv_iterations = inner_adv_iterations
        self.substitute_training_interval = substitute_training_interval
        self.label_type = label_type
        self.step_size = step_size
        self.top_k = 1
        self.substitute_model_load_dir = substitute_model_load_dir
        self.inference()

    def inference(self):
        self.malware_label = tf.constant([0., 1.])
        self.x_adv_tmp = tf.placeholder(dtype=tf.float32, shape=[None, DREBIN_FEATURE_Param['feature_dimension']])
        self.search_domain_TS = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        self.scaled_max_extended_TS = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        self.scaled_min_extended_TS  = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        self.iterate_attack_once = self.single_iteration()
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, DREBIN_FEATURE_Param['feature_dimension']])

    def single_iteration(self):
        output_logits = self.substitute_model.get_output_logits(self.x_adv_tmp)

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=output_logits,
                                                    labels=self.malware_label)
        )
        grad = tf.gradients(loss, self.x_adv_tmp)[0]
        abs_grad = tf.reshape(tf.abs(grad), (-1, self.input_dim))
        threshold = 0.

        tmp_increase_domain = tf.reshape(tf.less(self.x_adv_tmp, self.scaled_max_extended_TS), (-1, self.input_dim))
        tmp_increase_domain = tf.logical_and(tf.cast(self.search_domain_TS, tf.bool), tmp_increase_domain)
        tmp_domain1 = tf.logical_and(tf.greater(grad, tf.to_float(threshold)),
                                     tmp_increase_domain)

        tmp_decrease_domain = tf.reshape(tf.greater(self.x_adv_tmp, self.scaled_min_extended_TS), (-1, self.input_dim))
        tmp_decrease_domain = tf.logical_and(tf.cast(self.search_domain_TS, tf.bool), tmp_decrease_domain)
        tmp_domain2 = tf.logical_and(tf.less(grad, tf.to_float(-1 * threshold)),
                                     tmp_decrease_domain)

        tmp_search_domain = tf.cast(tf.logical_or(tmp_domain1, tmp_domain2), tf.float32)
        score_mask = tf.cast(abs_grad > 0., tf.float32) * tmp_search_domain

        abs_grad_mask = abs_grad * score_mask
        top_k_v, top_k_idx = tf.nn.top_k(abs_grad_mask, k=self.top_k)
        changed_pos = tf.reduce_sum(tf.one_hot(top_k_idx, depth=self.input_dim), axis=1)
        #changed_pos = tf.one_hot(top_k_idx, depth=self.input_dim)
        perturbations = tf.sign(grad) * changed_pos * tmp_search_domain
        # positions corresponds to the changed value will be neglected
        domain_out = self.search_domain_TS - changed_pos

        exist_modifiable_feature = (tf.reduce_sum(self.search_domain_TS, axis=1) >= 1)  # (?, )
        # 注意这里是以domain_in判断，而不是domain_out
        exist_modifiable_feature_float = tf.reshape(tf.cast(exist_modifiable_feature, tf.float32),
                                                    shape=[-1, 1])  # (?, 1)
        to_mod = perturbations * exist_modifiable_feature_float
        to_mod_reshape = tf.reshape(
            to_mod, shape=([-1] + self.x_adv_tmp.shape[1:].as_list()))

        x_out = self.x_adv_tmp + to_mod_reshape * self.step_size / tf.maximum(
            tf.reduce_sum(to_mod_reshape, -1, keepdims=True), 1)  #########

        x_out = tf.clip_by_value(x_out,
                                 clip_value_min=self.scaled_min_extended_TS,
                                 clip_value_max=self.scaled_max_extended_TS)
        x_adv_discrete = utils.map_to_discrete_domain_TF(self.normalizer, x_out)
        predict = self.substitute_model.get_output_pred(x_adv_discrete)
        _attack_success = \
            tf.logical_not(tf.equal(predict, tf.argmax(self.malware_label, axis=-1, output_type=tf.int32)))[0]

        return x_out, _attack_success, domain_out

    def blackbox_attack(self, single_malware_vector,
                        target_sess, target_graph,
                        substitute_sess,
                        substitute_saver):
        """
        超过maximum_query_iterations就当是攻击失败了。
        """
        self.substitute_model.load_param(self.substitute_model_load_dir, substitute_sess, substitute_saver)
        finetune_feature_set = []
        finetune_label_set = []
        query_num = 0
        result_dict = {}
        scaled_max_extended_var = np.maximum(
            np.multiply(self.scaled_clip_max, self.insertion_perm_array) +
            np.multiply(self.scaled_clip_min, 1. - self.insertion_perm_array),
            single_malware_vector
        )
        scaled_min_extended_var = np.minimum(
            np.multiply(self.scaled_clip_min, self.removal_perm_array) +
            np.multiply(self.scaled_clip_max, 1. - self.removal_perm_array),
            single_malware_vector
        )
        increase_domain_init = np.reshape(single_malware_vector < scaled_max_extended_var,
                                     [-1, self.input_dim])
        decrease_domian_init = np.reshape(single_malware_vector > scaled_min_extended_var,
                                     [-1, self.input_dim])
        search_domain_init = np.logical_or(increase_domain_init, decrease_domian_init).astype(np.float32)
        search_domain = search_domain_init
        while query_num < self.maximum_query_iterations:
            x_adv = np.reshape(single_malware_vector, (1, -1)).copy()
            for _ in range(self.inner_adv_iterations):
                x_adv, attack_success, search_domain = \
                    substitute_sess.run(self.iterate_attack_once,
                                        feed_dict={self.x_adv_tmp: x_adv,
                                                   self.scaled_max_extended_TS: scaled_max_extended_var,
                                                   self.scaled_min_extended_TS: scaled_min_extended_var,
                                                   self.search_domain_TS: search_domain})
                if bool(attack_success) is True:
                    break
            x_adv_discrete = utils.map_to_discrete_domain(self.normalizer, x_adv)
            single_modified = np.sum(np.abs(x_adv_discrete[0] - single_malware_vector) > 1e-6)
            with target_graph.as_default():
                with target_sess.as_default():
                    if self.label_type is "soft_label":
                        target_logits = target_sess.run(self.target_model.soft_output_logits,
                                                 feed_dict={self.target_model.x_input: x_adv_discrete})
                    elif self.label_type is "hard_label":
                        target_logits = target_sess.run(self.target_model.hard_output_logits,
                                                 feed_dict={self.target_model.x_input: x_adv_discrete})
                    else:
                        raise ValueError("the label tyep must be \"soft_label\" or \"hard_label\"")
            if np.argmax(target_logits, axis=-1)[0] == 0:
                average_modified = np.sum(np.abs(x_adv_discrete[0] - single_malware_vector[0]) > 1e-6)
                result_dict['query_num'] = query_num
                result_dict['attack_success'] = True
                result_dict['modified_nums'] = average_modified
                print("attack success")
                return result_dict
            finetune_feature_set.append(x_adv_discrete.copy()[0])
            finetune_label_set.append(target_logits.copy()[0])
            query_num += 1

            if query_num == self.maximum_query_iterations:
                average_modified = np.sum(
                    np.abs(utils.map_to_discrete_domain(self.normalizer, x_adv)[0] - single_malware_vector) > 1e-6)
                result_dict['attack_success'] = False
                result_dict['query_num'] = query_num
                result_dict['modified_nums'] = average_modified
                print("attack fail")
                return result_dict

            if query_num % self.substitute_training_interval == 0:
                self.substitute_model.substitute_training(train_x=np.array(finetune_feature_set),
                                                          train_y=np.array(finetune_label_set),
                                                          sess=substitute_sess,
                                                          n_epochs=6,
                                                          batch_size=16)



def attack():
    advtraining_number = 9
    substitute_training_number = 0
    advtraining_method = "pgdl2"
    target_model_name = 'target_model'
    substitute_model_name = "model_A"
    malware_dataset_name = "virustotal_2018_5M_17M"
    benware_dataset_name = "androzoo_benware_3M_17M"
    exp_name = "2_epochs_4096_" + malware_dataset_name + "_AND_" + benware_dataset_name
    adv_train_root = config.get("advtraining.drebin", "advtraining_drebin_root") + "/" + advtraining_method + \
                     "_" + exp_name
    ori_malware_feature_vectors = read_pickle(config.get(malware_dataset_name, "sample_vectors"))
    ori_malware_feature_labels = np.ones(len(ori_malware_feature_vectors))
    ori_malware_feature_labels = to_categorical(ori_malware_feature_labels, num_classes=2)
    if len(ori_malware_feature_vectors) > 200:
        exp_malware_feature_vectors, _ = \
            train_test_split(ori_malware_feature_vectors,
                             train_size=200, random_state=np.random.randint(low=0, high=999))
    else:
        exp_malware_feature_vectors = ori_malware_feature_vectors

    attack_success_record = []
    query_num_record = []
    modified_nums_record = []
    target_graph = tf.Graph()
    target_sess = tf.Session(graph=target_graph)
    substitute_graph = tf.Graph()
    substitute_sess = tf.Session(graph=substitute_graph)
    with target_graph.as_default():
        with target_sess.as_default():
            target_model = teacher_model(hyper_parameter=model_Param_dict[target_model_name],
                                         model_name=target_model_name,
                                         is_trainable=True)#只能选True了，不然optimizer那里会报错
            target_model_load_dir = adv_train_root + "/" + target_model_name + "/adv" + str(
                advtraining_number)
            target_saver = tf.train.Saver()
            target_sess.run(tf.global_variables_initializer())
            target_model.load_param(target_model_load_dir, target_sess, target_saver)
            print("target_model_accuracy: {}".format(
                target_sess.run(target_model.accuracy_output,
                                feed_dict={target_model.x_input: ori_malware_feature_vectors,
                                           target_model.y_input: ori_malware_feature_labels})
                                                    )
            )

    with substitute_graph.as_default():
        with substitute_sess.as_default():
            substitute_model_load_dir = adv_train_root + "/" + substitute_model_name + "/adv" + str(
                substitute_training_number)
            substitute_model = teacher_model(hyper_parameter=model_Param_dict[substitute_model_name],
                                             model_name=substitute_model_name,
                                             is_trainable=True)
            substitute_saver = tf.train.Saver()
            attacker = BlackBoxAttackPGDL1(
                target_model=target_model,
                substitute_model=substitute_model,
                substitute_model_load_dir=substitute_model_load_dir,
                label_type="soft_label",
                step_size=1.,
                warmup_iterations=16,
                maximum_query_iterations=100,
                inner_adv_iterations=100)
            substitute_sess.run(tf.global_variables_initializer())
    
    for single_feature_vector in tqdm.tqdm(exp_malware_feature_vectors):
        with substitute_graph.as_default():
            with substitute_sess.as_default():
                result_dict = attacker.blackbox_attack(single_feature_vector,
                                                       target_sess, target_graph,
                                                       substitute_sess,
                                                       substitute_saver)
        attack_success = result_dict['attack_success']
        query_num = result_dict['query_num']
        modified_nums = result_dict['modified_nums']
        attack_success_record.append(attack_success)
        query_num_record.append(query_num)
        modified_nums_record.append(modified_nums)
    utils.dump_pickle(np.array(attack_success_record),
                      config.get("advtraining.drebin", "advtraining_drebin_root") + "/attack_success_record_pgdl1_adv1")
    utils.dump_pickle(np.array(query_num_record),
                      config.get("advtraining.drebin", "advtraining_drebin_root") + "/query_num_record_pgdl1_adv1")
    utils.dump_pickle(np.array(modified_nums_record),
                      config.get("advtraining.drebin", "advtraining_drebin_root") + "/modified_nums_record_pgdl1_adv1")
    print("attack_success_rate: {:.2f}".format(np.mean(attack_success_record)))
    print("query_num: {:.2f}".format(np.mean(query_num_record)))

if __name__ == "__main__":
    attack()