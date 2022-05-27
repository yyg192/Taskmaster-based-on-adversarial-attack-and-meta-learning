import tensorflow as tf
from tools.DataProducer import DataProducer
import numpy as np
from keras.utils import to_categorical
import os
from timeit import default_timer
from tools.file_operation import read_pickle
from sklearn.model_selection import train_test_split
from tools import utils
from config import config
from config import model_Param_dict
import random
from advtraining_methods.pgdl2_generator import pgdl2_generator
from advtraining_methods.pgd_linfinity_generator import pgd_linfinity_generator
from advtraining_methods.pgdl1_generator import pgdl1_generator
from config import advtraining_methods
from config import advtraining_models
from config import DREBIN_FEATURE_Param

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def BASIC_DNN_GRAPH(
        x_input,
        name,
        hidden_neurons,
        output_dim=2,
        trainable=True,
        reuse=False):
    if len(hidden_neurons) == 1:
        with tf.variable_scope("{}".format(name), reuse=reuse):
            dense_in = tf.layers.dense(inputs=x_input, units=hidden_neurons[0], activation=tf.nn.relu,
                                       trainable=trainable, name="DENSE_IN_BACKBONE")
            dense_out = tf.layers.dense(inputs=dense_in, units=output_dim, activation=None,
                                        trainable=trainable, name="DENSE_OUT_BACKBONE")
    elif len(hidden_neurons) == 2:
        with tf.variable_scope("{}".format(name), reuse=reuse):
            dense_in = tf.layers.dense(inputs=x_input, units=hidden_neurons[0], activation=tf.nn.relu,
                                       trainable=trainable, name="DENSE_IN_BACKBONE")
            dense1 = tf.layers.dense(inputs=dense_in, units=hidden_neurons[1], activation=tf.nn.relu,
                                     trainable=trainable, name="DENSE1_BACKBONE")
            dense_out = tf.layers.dense(inputs=dense1, units=output_dim, activation=None,
                                        trainable=trainable, name="DENSE_OUT_BACKBONE")
    elif len(hidden_neurons) == 3:
        with tf.variable_scope("{}".format(name), reuse=reuse):
            """
            backbone
            """
            dense_in = tf.layers.dense(inputs=x_input, units=hidden_neurons[0], activation=tf.nn.relu,
                                       trainable=trainable, name="DENSE_IN_BACKBONE")
            dense1 = tf.layers.dense(inputs=dense_in, units=hidden_neurons[1], activation=tf.nn.relu,
                                     trainable=trainable, name="DENSE1_BACKBONE")
            dense2 = tf.layers.dense(inputs=dense1, units=hidden_neurons[2], activation=tf.nn.relu,
                                     trainable=trainable, name="DENSE2_BACKBONE")
            dense_out = tf.layers.dense(inputs=dense2, units=output_dim, activation=None,
                                        trainable=trainable, name="DENSE_OUT_BACKBONE")
    elif len(hidden_neurons) == 4:
        with tf.variable_scope("{}".format(name), reuse=reuse):
            """
            backbone
            """
            dense_in = tf.layers.dense(inputs=x_input, units=hidden_neurons[0], activation=tf.nn.relu,
                                       trainable=trainable, name="DENSE_IN_BACKBONE")
            dense1 = tf.layers.dense(inputs=dense_in, units=hidden_neurons[1], activation=tf.nn.relu,
                                     trainable=trainable, name="DENSE1_BACKBONE")
            dense2 = tf.layers.dense(inputs=dense1, units=hidden_neurons[2], activation=tf.nn.relu,
                                     trainable=trainable, name="DENSE2_BACKBONE")
            dense3 = tf.layers.dense(inputs=dense2, units=hidden_neurons[3], activation=tf.nn.relu,
                                     trainable=trainable, name="DENSE3_BACKBONE")
            dense_out = tf.layers.dense(inputs=dense3, units=output_dim, activation=None,
                                        trainable=trainable, name="DENSE_OUT_BACKBONE")
    elif len(hidden_neurons) == 5:
        with tf.variable_scope("{}".format(name), reuse=reuse):
            """
            backbone
            """
            dense_in = tf.layers.dense(inputs=x_input, units=hidden_neurons[0], activation=tf.nn.relu,
                                       trainable=trainable, name="DENSE_IN_BACKBONE")
            dense1 = tf.layers.dense(inputs=dense_in, units=hidden_neurons[1], activation=tf.nn.relu,
                                     trainable=trainable, name="DENSE1_BACKBONE")
            dense2 = tf.layers.dense(inputs=dense1, units=hidden_neurons[2], activation=tf.nn.relu,
                                     trainable=trainable, name="DENSE2_BACKBONE")
            dense3 = tf.layers.dense(inputs=dense2, units=hidden_neurons[3], activation=tf.nn.relu,
                                     trainable=trainable, name="DENSE3_BACKBONE")
            dense4 = tf.layers.dense(inputs=dense3, units=hidden_neurons[4], activation=tf.nn.relu,
                                     trainable=trainable, name="DENSE4_BACKBONE")
            dense_out = tf.layers.dense(inputs=dense4, units=output_dim, activation=None,
                                        trainable=trainable, name="DENSE_OUT_BACKBONE")
    else:
        raise ValueError("The layer of teacher model must less equal than 5 layers")
    return dense_out


class teacher_model():
    def __init__(self,
                 hyper_parameter,
                 model_name,
                 is_trainable=False,
                 reuse=False,
                 ):
        """
        变量接收区
        """
        self.model_name = model_name
        self.hp_params = utils.ParamWrapper(hyper_parameter)
        self.is_trainable = is_trainable
        """
        执行区
        """
        self.nn = BASIC_DNN_GRAPH  ##### 注意，我们可以通过修改graph函数来更改我们的模型 #####
        self.model_graph(reuse=reuse)
        self.global_train_step = tf.train.get_or_create_global_step()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer = \
                tf.train.AdamOptimizer(self.hp_params.learning_rate).minimize(self.output_cross_entropy,
                                                                              global_step=self.global_train_step)
    def model_graph(self, reuse=False):
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.hp_params.input_dim], name='X')
        self.y_input = tf.placeholder(dtype=tf.float32, shape=[None, self.hp_params.output_dim], name='Y')
        self.is_backbone_training = tf.placeholder(dtype=tf.bool, name="BACKBONE_TRAINING")  # 这个用于dropout和batchnorm的
        random_seed = np.random.randint(low=0, high=9999)
        tf.set_random_seed(random_seed)
        self.output_logits = self.nn(self.x_input,
                                     name=self.model_name,
                                     hidden_neurons=self.hp_params.hidden_neurons,
                                     output_dim=self.hp_params.output_dim,
                                     trainable=self.is_trainable,
                                     reuse=reuse)

        self.soft_output_logits = self.softmax_output_logits =  tf.nn.softmax(self.output_logits)
        self.hard_output_logits = tf.one_hot(tf.argmax(self.soft_output_logits, axis=1),
                                             self.hp_params.output_dim)

        self.output_cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=self.y_input,
                logits=self.output_logits
            )
        )

        self.y_pred_output = tf.argmax(self.output_logits, axis=1, output_type=tf.int32)

        self.accuracy_output = tf.reduce_mean(
            tf.to_float(tf.equal(self.y_pred_output, tf.argmax(self.y_input, axis=-1, output_type=tf.int32)))
        )

    def get_output_logits(self, x_tensor):
        fc_output_logits = self.nn(x_input=x_tensor,
                                   name=self.model_name,
                                   hidden_neurons=self.hp_params.hidden_neurons,
                                   output_dim=self.hp_params.output_dim,
                                   trainable=False,
                                   reuse=True)
        return fc_output_logits

    def get_output_pred(self, x_tensor):
        output_logits = self.get_output_logits(x_tensor)
        y_pred_output = tf.argmax(output_logits, axis=1, output_type=tf.int32)
        return y_pred_output

    def get_soft_output_logits(self, x_tensor):
        fc_output_logits = self.nn(x_input=x_tensor,
                                   name=self.model_name,
                                   hidden_neurons=self.hp_params.hidden_neurons,
                                   output_dim=self.hp_params.output_dim,
                                   trainable=False,
                                   reuse=True)
        softmax_output_logits = tf.nn.softmax(fc_output_logits)
        return softmax_output_logits

    def get_hard_output_logits(self, x_tensor):
        fc_output_logits = self.nn(x_input=x_tensor,
                                   name=self.model_name,
                                   hidden_neurons=self.hp_params.hidden_neurons,
                                   output_dim=self.hp_params.output_dim,
                                   trainable=False,
                                   reuse=True)
        hard_logits = tf.one_hot(tf.argmax(fc_output_logits, axis=1), self.hp_params.output_dim)
        return hard_logits

    def load_param(self, load_dir, sess, saver):
        if load_dir is None:
            raise ValueError("the load_dir is None, please check your code!")
        cur_checkpoint = tf.train.latest_checkpoint(load_dir)
        if cur_checkpoint is None:
            raise ValueError("the content of your checkpoint file is wrong!!!")
        #meta_dir = cur_checkpoint + ".meta"
        #restorer = tf.train.import_meta_graph(meta_dir)
        saver.restore(sess, cur_checkpoint)


    def substitute_training(self, train_x, train_y, sess, n_epochs, batch_size):
        train_input = DataProducer(train_x, train_y, batch_size, n_epochs=n_epochs)
        for step_idx, X_train_batch, Y_train_batch in train_input.next_batch():
            train_dict = {
                self.x_input: X_train_batch,
                self.y_input: Y_train_batch
            }
            sess.run(self.optimizer, feed_dict=train_dict)


    def train_backbone(self, train_x, train_y, valid_x, valid_y,
                       sess,
                       saver,
                       load_checkpoint_dir=None,
                       save_checkpoint_dir=None):
        train_input = DataProducer(train_x, train_y, self.hp_params.batch_size, n_epochs=self.hp_params.n_epochs)
        valid_input = DataProducer(valid_x, valid_y, self.hp_params.batch_size, n_epochs=self.hp_params.n_epochs)
        if load_checkpoint_dir is not None:
            cur_checkpoint = tf.train.latest_checkpoint(load_checkpoint_dir)
            saver.restore(sess, cur_checkpoint)
        training_time = 0.0
        train_input.reset_cursor()
        for step_idx, X_train_batch, Y_train_batch in train_input.next_batch():
            train_dict = {
                self.x_input: X_train_batch,
                self.y_input: Y_train_batch
            }
            start = default_timer()
            sess.run(self.optimizer, feed_dict=train_dict)
            end = default_timer()
            training_time = training_time + end - start
        valid_input.reset_cursor()
        valid_result_list = [sess.run([self.accuracy_output, self.y_pred_output],
                                      feed_dict={self.x_input: X_valid_batch,
                                                 self.y_input: Y_valid_batch
                                                 })
                             for [_, X_valid_batch, Y_valid_batch] in valid_input.next_batch()
                             ]
        valid_result = np.array(valid_result_list)
        _acc = np.mean(valid_result[:, 0])
        _pred_y = np.concatenate(valid_result[:, 1])
        from sklearn.metrics import f1_score
        _f1_score = f1_score(np.argmax(valid_y, axis=-1), _pred_y[:valid_y.shape[0]])
        print('    validation accuracy {:.5}%'.format(_acc * 100))
        print('    validation f1 score {:.5}%'.format(_f1_score * 100))
        if save_checkpoint_dir is not None:
            if os.path.exists(save_checkpoint_dir) is None:
                os.makedirs(save_checkpoint_dir)
            saver.save(sess, os.path.join(save_checkpoint_dir, 'checkpoint'), global_step=self.global_train_step)
            print("save backbone model")
        else:
            print("without saving the backbone model")


def _main():
    """
    model_name: model_A model_B model_C model_D model_E model_F model_G target_model
    """
    """
    model_name: model_1_160_Param ... model_15_160_Param
    """
    malware_dataset_name = "virustotal_2018_5M_17M"
    benware_dataset_name = "androzoo_benware_3M_17M"

    ori_malware_features_vectors = read_pickle(config.get(malware_dataset_name, "sample_vectors"))
    ori_benware_features_vectors = read_pickle(config.get(benware_dataset_name, "sample_vectors"))
    min_len = min(len(ori_malware_features_vectors), len(ori_benware_features_vectors)) - 1
    ori_malware_features_vectors, _ = \
        train_test_split(ori_malware_features_vectors,
                         train_size=min_len-1,
                         random_state=np.random.randint(0, 999))
    ori_benware_features_vectors, _ = \
        train_test_split(ori_benware_features_vectors,
                         train_size=min_len-1,
                         random_state=np.random.randint(0, 999))

    ori_malware_features_labels = to_categorical(np.ones(min_len-1), num_classes=2)
    ori_benware_features_labels = to_categorical(np.zeros(min_len-1), num_classes=2)

    ##########################################################
    adv_training_iteration_nums = 1
    advtraining_method = "pgdl2"
    models_name = advtraining_models
    exp_name = "20_epochs_4096_"+malware_dataset_name + "_AND_" + benware_dataset_name
    adv_train_root = config.get("advtraining.drebin", "advtraining_drebin_root") + "/" + advtraining_method+"_"+exp_name
    adv_samples_root = config.get("advtraining.drebin", "advsamples_drebin_root") + "/" + advtraining_method+"_"+exp_name
    if os.path.exists(adv_train_root) is False:
        os.makedirs(adv_train_root)
    if os.path.exists(adv_samples_root) is False:
        os.makedirs(adv_samples_root)
    ##########################################################
    for model_name in models_name:
        cur_model_graph = tf.Graph()
        cur_model_sess = tf.Session(graph=cur_model_graph)
        with cur_model_graph.as_default():
            with cur_model_sess.as_default():
                teacher = teacher_model(hyper_parameter=model_Param_dict[model_name],
                                        model_name=model_name,
                                        is_trainable=True)
                cur_model_saver = tf.train.Saver()
                if advtraining_method is "pgdl2":
                    attacker = pgdl2_generator(target_model=teacher,
                                               maximum_iterations=200,
                                               force_iteration=False,
                                               use_search_domain=True,
                                               random_mask=False,
                                               mask_rate=0.,
                                               step_size=10.)
                elif advtraining_method is "pgd_linfinity":  # 不打算要这个，这个会一次改变很多的特征
                    attacker = pgd_linfinity_generator(target_model=teacher,
                                                       maximum_iterations=100,
                                                       force_iteration=False,
                                                       use_search_domain=True,
                                                       random_mask=False,
                                                       mask_rate=0.,
                                                       step_size=0.2
                                                       )
                elif advtraining_method is "pgdl1":
                    attacker = pgdl1_generator(target_model=teacher,
                                               maximum_iterations=100,
                                               force_iteration=False,
                                               use_search_domain=True,
                                               random_mask=False,
                                               mask_rate=0.,
                                               top_k=1,
                                               step_size=1.
                                               )
                elif advtraining_method is "jsma":
                    raise NotImplementedError("jsma is unfinished yet")
                else:
                    raise NotImplementedError("other adv training method is not supported yet")

                cur_model_sess.run(tf.global_variables_initializer())

        for i in range(adv_training_iteration_nums):
            print("####################### adv_training_iteration_nums: {} #######################".format(i))
            random_state = np.random.randint(low=0, high=999)
            if i == 0:
                all_malware_features_vectors = ori_malware_features_vectors.copy()
                all_benware_features_vectors = ori_benware_features_vectors.copy()
                all_malware_features_labels = to_categorical(np.ones(len(all_malware_features_vectors)), num_classes=2)
                all_benware_features_labels = to_categorical(np.zeros(len(all_benware_features_vectors)), num_classes=2)
                print("all_malware_features_vectors.shape: {} all_benware_features_vectors.shape: {}".format(
                    all_malware_features_vectors.shape,
                    all_benware_features_vectors.shape))
                all_features_vectors = np.concatenate((all_malware_features_vectors, all_benware_features_vectors),
                                                      axis=0)
                all_features_labels = np.concatenate((all_malware_features_labels, all_benware_features_labels),
                                                     axis=0)
            else:
                all_malware_features_vectors = ori_malware_features_vectors.copy()
                all_benware_features_vectors = ori_benware_features_vectors.copy()
                for j in range(i):
                    ADV_J_SAMPLES_LOAD_DIR = adv_samples_root + "/" + model_name + "/adv" + str(j)
                    single_adv_malware_features_vectors = read_pickle(ADV_J_SAMPLES_LOAD_DIR + "/x_adv_success_malware")
                    single_adv_benware_features_vectors = read_pickle(ADV_J_SAMPLES_LOAD_DIR + "/x_adv_success_benware")
                    if single_adv_malware_features_vectors > 1000:
                        single_adv_malware_features_vectors, _ = train_test_split(single_adv_malware_features_vectors,
                                                                                  train_size=1000,
                                                                                  random_state=np.random.randint(0, 999))
                    if single_adv_benware_features_vectors > 1000:
                        single_adv_benware_features_vectors , _ = train_test_split(single_adv_benware_features_vectors,
                                                                                   train_size=1000,
                                                                                   random_state=np.random.randint(0, 999))

                    all_malware_features_vectors = np.concatenate(
                        (all_malware_features_vectors, single_adv_malware_features_vectors), axis=0)
                    all_benware_features_vectors = np.concatenate(
                        (all_benware_features_vectors, single_adv_benware_features_vectors), axis=0)

                all_malware_features_vectors = np.array(all_malware_features_vectors)
                all_benware_features_vectors = np.array(all_benware_features_vectors)

                #all_malware_features_vectors, _ = train_test_split(all_malware_features_vectors, train_size=min_len-1)
                #all_benware_features_vectors, _ = train_test_split(all_benware_features_vectors, train_size=min_len-1)
                all_malware_features_labels = to_categorical(np.ones(len(all_malware_features_vectors)), num_classes=2)
                all_benware_features_labels = to_categorical(np.zeros(len(all_benware_features_vectors)), num_classes=2)
                all_features_vectors = np.concatenate((all_malware_features_vectors, all_benware_features_vectors),
                                                      axis=0)
                all_features_labels = np.concatenate((all_malware_features_labels, all_benware_features_labels),
                                                     axis=0)
                print("all_malware_features_vectors.shape: {} all_benware_features_vectors.shape: {}".format(
                    all_malware_features_vectors.shape,
                    all_benware_features_vectors.shape))

            ADV_CHECKPOINT_SAVE_DIR = adv_train_root + "/" + model_name + "/adv" + str(i)
            if os.path.exists(ADV_CHECKPOINT_SAVE_DIR) is False:
                os.makedirs(ADV_CHECKPOINT_SAVE_DIR)

            train_features, valid_features, train_y, valid_y = \
                train_test_split(all_features_vectors, all_features_labels, test_size=0.1, random_state=random_state)

            print("train_features.shape: {}".format(train_features.shape))
            print("train_y.shape: {}".format(train_y.shape))
            ADV_SAMPLES_SAVE_DIR = adv_samples_root + "/" + model_name + "/adv" + str(i)
            if os.path.exists(ADV_SAMPLES_SAVE_DIR) is False:
                os.makedirs(ADV_SAMPLES_SAVE_DIR)
            with cur_model_graph.as_default():
                with cur_model_sess.as_default():
                    teacher.train_backbone(train_features, train_y, valid_features, valid_y,
                                           cur_model_sess,
                                           saver=cur_model_saver,
                                           load_checkpoint_dir=None,
                                           save_checkpoint_dir=ADV_CHECKPOINT_SAVE_DIR)
                    print("preprocessing malware")
                    attacker.generate_attack_samples_teacher(ori_malware_features_vectors,
                                                             ori_malware_features_labels,
                                                             ADV_SAMPLES_SAVE_DIR,
                                                             cur_model_sess,
                                                             "malware")
                    print("preprocessing benware")
                    attacker.generate_attack_samples_teacher(ori_benware_features_vectors,
                                                             ori_benware_features_labels,
                                                             ADV_SAMPLES_SAVE_DIR,
                                                             cur_model_sess,
                                                             "benware")
        cur_model_sess.close()
        tf.reset_default_graph()


if __name__ == "__main__":
    _main()
