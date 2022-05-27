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


class BlackBoxAttackPGDLInfinity(AttackerBase):
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
        super(BlackBoxAttackPGDLInfinity, self).__init__()
        self.target_model = target_model
        self.substitute_model = substitute_model
        self.warmup_iterations = warmup_iterations
        self.maximum_query_iterations = maximum_query_iterations
        self.inner_adv_iterations = inner_adv_iterations
        self.substitute_training_interval = substitute_training_interval
        self.label_type = label_type
        self.malware_label = tf.constant([0., 1.])
        self.step_size = step_size
        self.substitute_model_load_dir = substitute_model_load_dir
        self.inference()

    def inference(self):
        self.x_adv_tmp = tf.placeholder(dtype=tf.float32, shape=[None, DREBIN_FEATURE_Param['feature_dimension']])
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, DREBIN_FEATURE_Param['feature_dimension']])
        self.scaled_max_extended_TS = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        self.scaled_min_extended_TS = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])
        self.iterate_attack_once = self.single_iteration()

    def single_iteration(self):
        logits = self.substitute_model.get_output_logits(self.x_adv_tmp)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                      labels=self.malware_label))  # average loss, may cause leakage issue
        grad = tf.gradients(loss, self.x_adv_tmp)[0]
        perturbations = utils.optimize_linear(grad, tf.to_float(self.step_size))
        x_adv_out = self.x_adv_tmp + perturbations
        x_adv_out = tf.clip_by_value(x_adv_out,
                                     clip_value_min=self.scaled_min_extended_TS,
                                     clip_value_max=self.scaled_max_extended_TS)
        x_adv_discrete = utils.map_to_discrete_domain_TF(self.normalizer, x_adv_out)
        predict = self.substitute_model.get_output_pred(x_adv_discrete)
        attack_success = \
        tf.logical_not(tf.equal(predict, tf.argmax(self.malware_label, axis=-1, output_type=tf.int32)))[0]
        return x_adv_out, logits, attack_success

    def blackbox_attack(self, single_malware_vector,
                        target_sess, target_graph,
                        substitute_sess,
                        substitute_saver):
        """
        超过maximum_query_iterations就当是攻击失败了。
        """
        self.substitute_model.load_param(self.substitute_model_load_dir, substitute_sess, substitute_saver)
        x_adv = np.reshape(single_malware_vector, (1, -1)).copy()
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
        while query_num < self.maximum_query_iterations:
            for _ in range(self.inner_adv_iterations):
                x_adv, logits, attack_success = \
                    substitute_sess.run(self.iterate_attack_once,
                                        feed_dict={self.x_adv_tmp: x_adv,
                                                   self.scaled_max_extended_TS: scaled_max_extended_var,
                                                   self.scaled_min_extended_TS: scaled_min_extended_var})
                if bool(attack_success) is True:
                    break
            x_adv_discrete = utils.map_to_discrete_domain(self.normalizer, x_adv)
            single_modified = np.sum(np.abs(x_adv_discrete[0] - single_malware_vector[0]) > 1e-6)
            print("query_num: {} single_modified: {}".format(query_num, single_modified))
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
                average_modified = np.sum(np.abs(x_adv_discrete[0] - single_malware_vector) > 1e-6)
                result_dict['query_num'] = query_num
                result_dict['attack_success'] = True
                result_dict['modified_nums'] = average_modified
                return result_dict
            finetune_feature_set.append(x_adv.copy()[0])
            finetune_label_set.append(target_logits[0])
            query_num += 1
            if query_num % self.substitute_training_interval == 0:
                self.substitute_model.substitute_training(train_x=np.array(finetune_feature_set),
                                                          train_y=np.array(finetune_label_set),
                                                          sess=substitute_sess,
                                                          n_epochs=1,
                                                          batch_size=1)
        average_modified = np.sum(np.abs(utils.map_to_discrete_domain(self.normalizer, x_adv)[0] - single_malware_vector[0]) > 1e-6)
        result_dict['attack_success'] = False
        result_dict['query_num'] = query_num
        result_dict['modified_nums'] = average_modified
        return result_dict


def attack():
    advtraining_number = 3
    advtraining_method = "pgdl2"
    target_model_name = 'target_model'
    substitute_model_name = "model_A"
    ori_malware_feature_vectors = read_pickle(config.get('feature_preprocess.drebin', 'malware_sample_vectors'))
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
                                         is_trainable=True)  # 只能选True了，不然optimizer那里会报错
            target_model_load_dir = config.get("advtraining.drebin", "advtraining_drebin_root") + \
                                    "/" + advtraining_method + "/" + target_model_name + "/adv" + str(
                advtraining_number)
            target_saver = tf.train.Saver()
            target_sess.run(tf.global_variables_initializer())
            target_model.load_param(target_model_load_dir, target_sess, target_saver)

    with substitute_graph.as_default():
        with substitute_sess.as_default():
            substitute_model_load_dir = config.get("advtraining.drebin", "advtraining_drebin_root") + \
                                        "/" + advtraining_method + "/" + substitute_model_name + "/adv" + str(
                advtraining_number)
            substitute_model = teacher_model(hyper_parameter=model_Param_dict[substitute_model_name],
                                             model_name=substitute_model_name,
                                             is_trainable=True)
            substitute_saver = tf.train.Saver()
            attacker = BlackBoxAttackPGDLInfinity(
                target_model=target_model,
                substitute_model=substitute_model,
                substitute_model_load_dir=substitute_model_load_dir,
                label_type="soft_label",
                step_size=0.1,
                warmup_iterations=16,
                maximum_query_iterations=200,
                inner_adv_iterations=20)
            substitute_sess.run(tf.global_variables_initializer())
    for single_feature_vector in tqdm.tqdm(exp_malware_feature_vectors):
        with substitute_graph.as_default():
            with substitute_sess.as_default():
                result_dict = attacker.blackbox_attack(single_feature_vector,
                                                       target_sess,
                                                       target_graph,
                                                       substitute_sess,
                                                       substitute_saver)
        attack_success = result_dict['attack_success']
        query_num = result_dict['query_num']
        modified_nums = result_dict['modified_nums']
        attack_success_record.append(attack_success)
        query_num_record.append(query_num)
        modified_nums_record.append(modified_nums)
    utils.dump_pickle(np.array(attack_success_record),
                      config.get("advtraining.drebin", "advtraining_drebin_root") + "/attack_success_record")
    utils.dump_pickle(np.array(query_num_record),
                      config.get("advtraining.drebin", "advtraining_drebin_root") + "/query_num_record")
    utils.dump_pickle(np.array(modified_nums_record),
                      config.get("advtraining.drebin", "advtraining_drebin_root") + "/modified_nums_record")

    print("attack_success_rate: {:.2f}".format(np.mean(attack_success_record)))
    # print("query_num: {:.2f}".format(np.mean(query_num_record)))


if __name__ == "__main__":
    attack()
