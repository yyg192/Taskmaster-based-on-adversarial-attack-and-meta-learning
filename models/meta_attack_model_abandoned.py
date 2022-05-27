import sklearn.utils
import tensorflow as tf
from config import config
from config import modelSimulator_Param
from config import model_Param_dict
from config import model_160_Param
from tools.utils import ParamWrapper
from sklearn.model_selection import train_test_split
from models.teacher_model import teacher_model
from advtraining_methods.pgdl2_generator import pgdl2_generator
from advtraining_methods.pgdl1_generator import pgdl1_generator
from advtraining_methods.pgd_linfinity_generator import pgd_linfinity_generator
from tools.utils import read_pickle
from tools.DataProducer import DataProducer
from tools.utils import dump_pickle
from config import advtraining_methods
from keras.utils import to_categorical
import numpy as np
import os
import tqdm
import warnings

warnings.filterwarnings('ignore')


class MetaModel:
    def __init__(self,
                 advtraining_number=None,
                 advtraining_method=None,
                 distillation_method=None,
                 exp_name=None):
        """ task数量 """
        self.K_task_nums = 32
        """ support_shot 和 query_shot """
        self.meta_train_t = 30  # 这个不能给太高，给到50的话loss就很高
        self.meta_test_t = 30
        """ 模型保存间隔 """
        self.save_interval = 1
        self.print_summary_interval = 1
        """测试间隔"""
        self.test_interval = 1
        """ 内外迭代次数"""
        self.inner_update_iterations = 10
        self.outter_update_iterations = 20
        """  学习率  """
        self.inner_update_learning_rate = 1e-5  # 试一下0.01
        self.outter_update_learning_rate = 1e-5
        """ 使用什么类型的损失值 可选的有cross_entropy:"CE" 和 mean_square_error:"MSE" """
        self.loss = "CE"  # option: "CE" 、 "MSE" 、
        """ 使用的教师模型以及被攻击的模型经历过多少次对抗训练"""
        self.adv_training_number = advtraining_number
        self.distillation_method = distillation_method
        if advtraining_method is not None and advtraining_number is not None and exp_name is not None:
            """ 模型保存地址 """
            self.simulator_model_save_dir = \
                config.get("advtraining.drebin", "advtraining_drebin_root") + \
                "/" + advtraining_method + "/meta_simulator" + '/' + exp_name + "/adv" + str(advtraining_number)
            print("simulator_model_save_dir: ", self.simulator_model_save_dir)
            if os.path.exists(self.simulator_model_save_dir) is False:
                os.makedirs(self.simulator_model_save_dir)
            """ tensorboard log地址 """
            self.tensorboard_log_dir = \
                config.get("metatraining.drebin", "model_simulator_log_dir") + '/' + exp_name + "/" + str(
                    advtraining_method) \
                + "/adv" + str(advtraining_number)
            if os.path.exists(self.tensorboard_log_dir) is False:
                os.makedirs(self.tensorboard_log_dir)
        else:
            self.simulator_model_save_dir = None
            self.tensorboard_log_dir = None
        """接收可用参数"""
        self.param = ParamWrapper(modelSimulator_Param)

        """为当前模型定义一个sess和graph"""
        self.weights = self.get_weights()
        self.support_xb_TS = tf.placeholder(dtype=tf.float32,
                                            shape=[self.K_task_nums, self.meta_train_t, self.param.input_dim])
        self.support_yb_TS = tf.placeholder(dtype=tf.float32,
                                            shape=[self.K_task_nums, self.meta_train_t, self.param.output_dim])
        self.query_xb_TS = tf.placeholder(dtype=tf.float32,
                                          shape=[self.K_task_nums, self.meta_test_t, self.param.input_dim])
        self.query_yb_TS = tf.placeholder(dtype=tf.float32,
                                          shape=[self.K_task_nums, self.meta_test_t, self.param.output_dim])

        if self.tensorboard_log_dir is not None:
            self.writter = tf.summary.FileWriter(self.tensorboard_log_dir)
        self.global_step = tf.train.get_or_create_global_step()
        self.build_regular_simulator()
        self.build_meta_simulator()
        if self.tensorboard_log_dir is not None:
            self.summary_op = tf.summary.merge_all()

        # self.simulator_sess.run(tf.global_variables_initializer())

        """
        model_name   random_seed   hidden_neurons
        input_dim output_dim   warmup_n_epochs   distillation_n_epochs
        batch_size   learning_rate   optimizer
        """

    def get_output_logits(self, x_tensor):
        output_logits = self.forward(x_tensor, self.weights)
        return output_logits

    def build_regular_simulator(self):
        self.x_input = tf.placeholder(dtype=tf.float32,
                                      shape=[None, self.param.input_dim])
        self.y_input = tf.placeholder(dtype=tf.float32,
                                      shape=[None, self.param.output_dim])
        self.regular_logits = self.forward(self.x_input, self.weights)
        self.regular_preds = tf.argmax(self.regular_logits, axis=1, output_type=tf.int32)
        self.regular_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.regular_logits,
                                                    labels=self.y_input)
        )
        self.regular_accuracy = tf.reduce_mean(
            tf.to_float(tf.equal(self.regular_preds,
                                 tf.argmax(self.y_input, axis=-1, output_type=tf.int32)))
        )

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.regular_optimizer = tf.train.AdamOptimizer(self.param.warmup_learning_rate).minimize(self.regular_loss)

    def load_param(self, load_dir, sess):
        if load_dir is None:
            raise ValueError("the load dir is none, please check your code")
        cur_checkpoint = tf.train.latest_checkpoint(load_dir)
        meta_dir = cur_checkpoint + ".meta"
        restorer = tf.train.import_meta_graph(meta_dir)
        restorer.restore(sess, cur_checkpoint)
        print("load checkpoint_dir: ", cur_checkpoint)
        return restorer

    def warmup_training(self, train_x, valid_x, train_y, valid_y, sess):
        train_input = DataProducer(train_x, train_y, self.param.batch_size,
                                   n_epochs=self.param.warmup_n_epochs)
        valid_input = DataProducer(valid_x, valid_y, self.param.batch_size,
                                   n_epochs=self.param.warmup_n_epochs)

        train_input.reset_cursor()
        output_steps = 100
        for step_idx, X_train_batch, Y_train_batch in train_input.next_batch():
            train_dict = {
                self.x_input: X_train_batch,
                self.y_input: Y_train_batch
            }
            if (step_idx + 1) % output_steps == 0:
                valid_input.reset_cursor()
                valid_result_list = [sess.run([self.regular_accuracy, self.regular_preds],
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
            sess.run(self.regular_optimizer, feed_dict=train_dict)

    def build_meta_simulator(self, mode='train'):
        """
        support_xb: [self.K_task_nums, self.meta_train_t, 1024]
        support_yb: [self.K_task_nums, self.meta_train_t, 2]
        query_xb:   [self.K_task_nums, self.meta_test_t, 1024]
        query_yb:   [self.K_task_nums, self.meta_test_t, 2]
        """

        def meta_task(input):
            """
            support_x: [self.meta_train_t, 1024]
            support_y: [self.meta_train_t, 2]
            query_x:   [self.meta_test_t, 1024]
            query_y:   [self.meta_test_t, 2]
            """
            # 处理一个meta_task
            support_x, support_y, query_x, query_y = input

            # support_x.shape=(self.meta_train_t, self.input_dim) check!!!
            # support_y.shape=(self.meta_train_t, self.output_dim)
            # to record the op in t update step.
            support_losses, support_preds, support_accs, query_preds, query_losses, query_accs = [], [], [], [], [], []
            support_pred = self.forward(support_x, self.weights)
            # support_pred.shape=(10, 2)
            support_loss = tf.nn.softmax_cross_entropy_with_logits(logits=support_pred, labels=support_y)
            # support_loss.shape=(10, )
            support_acc = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(support_pred, dim=1), axis=1),
                                                      tf.argmax(support_y, axis=1))
            support_losses.append(support_loss)
            support_preds.append(support_pred)
            support_accs.append(support_acc)
            # support_acc.shape=()
            # compute gradients
            grads = tf.gradients(support_loss, list(self.weights.values()))
            # 这里的grads就只有一份，并不是每一个loss对应每一个sample有一个grad，不知道为什么，但似乎就是这样的。
            # grad and variable dict
            gvs = dict(zip(self.weights.keys(), grads))

            # theta_pi = theta - alpha * grads
            fast_weights = dict(zip(self.weights.keys(),
                                    [self.weights[key] - self.inner_update_learning_rate * gvs[key] for key in
                                     self.weights.keys()]))
            # use theta_pi to forward meta-test
            query_pred = self.forward(query_x, fast_weights)
            # meta-test loss
            query_loss = tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=query_y)
            # record T0 pred and loss for meta-test
            query_preds.append(query_pred)
            query_losses.append(query_loss)

            for _ in range(1, self.inner_update_iterations):
                pred = self.forward(support_x, fast_weights)
                loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred,
                                                               labels=support_y)
                acc = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(pred, dim=1), axis=1),
                                                  tf.argmax(support_y, axis=1))
                grads = tf.gradients(loss, list(fast_weights.values()))
                # compose grad and variable dict
                gvs = dict(zip(fast_weights.keys(), grads))
                # update theta_pi according to varibles
                fast_weights = dict(
                    zip(fast_weights.keys(), [fast_weights[key] - self.inner_update_learning_rate * gvs[key]
                                              for key in fast_weights.keys()]))
                support_losses.append(loss)
                support_preds.append(pred)
                support_accs.append(acc)
                # forward on theta_pi
                query_pred = self.forward(query_x, fast_weights)
                # we need accumulate all meta-test losses to update theta
                query_loss = tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=query_y)
                query_preds.append(query_pred)
                query_losses.append(query_loss)

            for i in range(self.inner_update_iterations):
                query_accs.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(query_preds[i], dim=1), axis=1),
                                                              tf.argmax(query_y, axis=1)))
            result = [support_preds, support_losses, support_accs, query_preds, query_losses, query_accs]
            # support_pred.shape=(self.meta_train_t,2)  support_loss.shape=(self.meta_train_t, )  support_acc.shape=()
            # query_preds.shape=(self.T_fast_weights_iterations, self.meta_test_t, 2)
            # query_losses.shape=(self.T_fast_weights_iterations, self.meta_test_t, )
            return result

        #             support_pred,                                 support_loss,
        out_dtype = [[tf.float32] * self.inner_update_iterations, [tf.float32] * self.inner_update_iterations,
                     #             support_acc
                     [tf.float32] * self.inner_update_iterations,
                     #           query_preds,                                   query_losses
                     [tf.float32] * self.inner_update_iterations, [tf.float32] * self.inner_update_iterations,
                     #           query_accs
                     [tf.float32] * self.inner_update_iterations]
        result = tf.map_fn(meta_task,
                           elems=(self.support_xb_TS, self.support_yb_TS, self.query_xb_TS, self.query_yb_TS),
                           dtype=out_dtype,
                           parallel_iterations=self.K_task_nums,
                           name='map_fn')
        support_preds_tasks, support_losses_tasks, support_accs_tasks, \
        query_preds_tasks, query_losses_tasks, query_accs_tasks = result

        if mode is 'train':
            # average loss
            self.support_loss = support_loss = [tf.reduce_sum(support_losses_tasks[j]) / self.K_task_nums
                                                for j in range(self.inner_update_iterations)]
            # [avgloss_t1, avgloss_t2, ..., avgloss_K]
            self.query_losses = query_losses = [tf.reduce_sum(query_losses_tasks[j]) / self.K_task_nums
                                                for j in range(self.inner_update_iterations)]
            # average accuracy
            self.support_acc = support_acc = [tf.reduce_sum(support_accs_tasks[j]) / self.K_task_nums
                                              for j in range(self.inner_update_iterations)]
            # average accuracies
            self.query_accs = query_accs = [tf.reduce_sum(query_accs_tasks[j]) / self.K_task_nums
                                            for j in range(self.inner_update_iterations)]
            # 如果你有batch_normalization的话！！！ 一定要记得with control_dependencies，不过这里没有
            # # add batch_norm ops before meta_op
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
            # 	# TODO: the update_ops must be put before tf.train.AdamOptimizer,
            # 	# otherwise it throws Not in same Frame Error.
            # 	meta_loss = tf.identity(self.query_losses[-1])

            # meta-train optim
            optimizer = tf.train.AdamOptimizer(self.inner_update_learning_rate, name='meta_optim')
            # meta-train gradients, query_losses[-1] is the accumulated loss across over tasks.
            gvs = optimizer.compute_gradients(self.query_losses[-1])
            # meta-train grads clipping
            gvs = [(tf.clip_by_norm(grad, 10), var) for grad, var in gvs]
            # update theta
            self.meta_op = optimizer.apply_gradients(gvs, global_step=self.global_step)
        else:  # test & eval

            # average loss
            self.test_support_loss = support_loss = [tf.reduce_sum(support_losses_tasks[j]) / self.K_task_nums
                                                     for j in range(self.inner_update_iterations)]
            # [avgloss_t1, avgloss_t2, ..., avgloss_K]
            self.test_query_losses = query_losses = [tf.reduce_sum(query_losses_tasks[j]) / self.K_task_nums
                                                     for j in range(self.inner_update_iterations)]
            # average accuracy
            self.test_support_acc = support_acc = [tf.reduce_sum(support_accs_tasks[j]) / self.K_task_nums
                                                   for j in range(self.inner_update_iterations)]
            # average accuracies
            self.test_query_accs = query_accs = [tf.reduce_sum(query_accs_tasks[j]) / self.K_task_nums
                                                 for j in range(self.inner_update_iterations)]
        if self.tensorboard_log_dir is not None:
            # NOTICE: every time build model, support_loss will be added to the summary, but it's different.
            tf.summary.scalar(mode + '：support loss', support_loss[-1])
            tf.summary.scalar(mode + '：support acc', support_acc[-1])
            # for j in range(self.inner_update_iterations):
            tf.summary.scalar(mode + '：query loss', query_losses[-1])
            tf.summary.scalar(mode + '：query acc ', query_accs[-1])

    def meta_training(self, teachers_info,
                      attack_feature_vectors,
                      attack_feature_labels,
                      sess,
                      simulator_saver):
        teacher_models = teachers_info['models']
        teacher_models_sess = teachers_info['sess']
        teacher_models_graph = teachers_info['graph']
        for iter in range(self.outter_update_iterations):
            # sample K samples x1,...,xK from attack_feature_vectors
            K_feature_vectors, _, K_feature_labels, _ = train_test_split(attack_feature_vectors,
                                                                         attack_feature_labels,
                                                                         train_size=self.K_task_nums * (
                                                                                     self.meta_train_t + self.meta_test_t),
                                                                         random_state=np.random.randint(0, 999))
            K_support_x_set = []
            K_support_y_set = []
            K_query_x_set = []
            K_query_y_set = []
            for k in tqdm.tqdm(range(self.K_task_nums)):
                random_teacher_idx = np.random.randint(0, len(teacher_models))
                random_teacher = teacher_models[random_teacher_idx]
                random_teacher_sess = teacher_models_sess[random_teacher_idx]
                random_teacher_graph = teacher_models_graph[random_teacher_idx]
                with random_teacher_graph.as_default():
                    with random_teacher_sess.as_default():
                        support_x = K_feature_vectors[(k * (self.meta_train_t + self.meta_test_t)):(
                                    k * (self.meta_train_t + self.meta_test_t) + self.meta_train_t)]
                        query_x = K_feature_vectors[(k * (self.meta_train_t + self.meta_test_t) + self.meta_train_t):(
                                    (k + 1) * (self.meta_train_t + self.meta_test_t))]
                        support_y = random_teacher_sess.run(random_teacher.softmax_output_logits,
                                                            feed_dict={random_teacher.x_input: support_x})
                        query_y = random_teacher_sess.run(random_teacher.softmax_output_logits,
                                                          feed_dict={random_teacher.x_input: query_x})
                        K_support_x_set.append(support_x)
                        K_support_y_set.append(support_y)
                        K_query_x_set.append(query_x)
                        K_query_y_set.append(query_y)
            K_support_x_set = np.array(K_support_x_set)
            K_support_y_set = np.array(K_support_y_set)
            K_query_x_set = np.array(K_query_x_set)
            K_query_y_set = np.array(K_query_y_set)

            ops = [self.meta_op]

            # add summary and print op
            sess.run(ops,
                     feed_dict={self.support_xb_TS: K_support_x_set,
                                self.support_yb_TS: K_support_y_set,
                                self.query_xb_TS: K_query_x_set,
                                self.query_yb_TS: K_query_y_set})
            if self.tensorboard_log_dir is not None and iter % self.print_summary_interval == 0:
                summary = sess.run(self.summary_op,
                                   feed_dict={self.support_xb_TS: K_support_x_set,
                                              self.support_yb_TS: K_support_y_set,
                                              self.query_xb_TS: K_query_x_set,
                                              self.query_yb_TS: K_query_y_set})
                self.writter.add_summary(summary, iter)

            if iter % self.save_interval == 0 and iter != 0:
                if self.simulator_model_save_dir is not None:
                    simulator_saver.save(sess, self.simulator_model_save_dir + "/checkpoint",
                                         global_step=self.global_step)
                    print("meta training simulator_model save dir: ".format(self.simulator_model_save_dir))
                else:
                    print("meta training simulator_model is not save")
            # shape: [self.K_task_nums, 50, 1024]

    def get_weights(self):
        weights = {}
        fc_initializer = tf.glorot_uniform_initializer()  # !!!!!!!!!!!!!!!
        with tf.variable_scope(self.param.model_name, reuse=tf.AUTO_REUSE):
            for i in range(1, len(self.param.hidden_neurons) + 1):
                w_name = 'fc' + str(i) + 'w'
                b_name = 'fc' + str(i) + 'b'
                w_key_name = 'w' + str(i)
                b_key_name = 'b' + str(i)
                if i == 1:
                    weights[w_key_name] = tf.get_variable(w_name,
                                                          [self.param.input_dim, self.param.hidden_neurons[i - 1]],
                                                          initializer=fc_initializer)
                    weights[b_key_name] = tf.get_variable(b_name,
                                                          initializer=tf.zeros([self.param.hidden_neurons[i - 1]]))
                else:
                    weights[w_key_name] = tf.get_variable(w_name,
                                                          [self.param.hidden_neurons[i - 2],
                                                           self.param.hidden_neurons[i - 1]],
                                                          initializer=fc_initializer)
                    weights[b_key_name] = tf.get_variable(b_name,
                                                          initializer=tf.zeros([self.param.hidden_neurons[i - 1]]))

                weights['wout'] = tf.get_variable('fcoutw',
                                                  [self.param.hidden_neurons[-1], self.param.output_dim],
                                                  initializer=fc_initializer)
                weights['bout'] = tf.get_variable('fcoutb',
                                                  initializer=tf.zeros([self.param.output_dim]))
        return weights

    def forward(self, x, weights):
        x = tf.reshape(x, [-1, self.param.input_dim], name='reshape1')
        if len(self.param.hidden_neurons) == 2:
            hidden1 = tf.add(tf.matmul(x, weights['w1']), weights['b1'], name='fc1')
            hidden1 = tf.nn.relu(hidden1)
            hidden2 = tf.add(tf.matmul(hidden1, weights['w2']), weights['b2'], name='fc2')
            hidden2 = tf.nn.relu(hidden2)
            output = tf.add(tf.matmul(hidden2, weights['wout']), weights['bout'], name='fcout')
        elif len(self.param.hidden_neurons) == 3:
            hidden1 = tf.add(tf.matmul(x, weights['w1']), weights['b1'], name='fc1')
            hidden1 = tf.nn.relu(hidden1)
            hidden2 = tf.add(tf.matmul(hidden1, weights['w2']), weights['b2'], name='fc2')
            hidden2 = tf.nn.relu(hidden2)
            hidden3 = tf.add(tf.matmul(hidden2, weights['w3']), weights['b3'], name='fc3')
            hidden3 = tf.nn.relu(hidden3)
            output = tf.add(tf.matmul(hidden3, weights['wout']), weights['bout'], name='fcout')
        elif len(self.param.hidden_neurons) == 4:
            hidden1 = tf.add(tf.matmul(x, weights['w1']), weights['b1'], name='fc1')
            hidden1 = tf.nn.relu(hidden1)
            hidden2 = tf.add(tf.matmul(hidden1, weights['w2']), weights['b2'], name='fc2')
            hidden2 = tf.nn.relu(hidden2)
            hidden3 = tf.add(tf.matmul(hidden2, weights['w3']), weights['b3'], name='fc3')
            hidden3 = tf.nn.relu(hidden3)
            hidden4 = tf.add(tf.matmul(hidden3, weights['w4']), weights['b4'], name='fc4')
            hidden4 = tf.nn.relu(hidden4)
            output = tf.add(tf.matmul(hidden4, weights['wout']), weights['bout'], name='fcout')
        else:
            raise NotImplementedError("forward not implemented")
        return output


def _main():
    # TEACHERS = ['model_A', 'model_B', 'model_C', 'model_D', 'model_E', 'model_F', 'model_G']
    teachers_name = ['model_' + str(i) + "_160x3" for i in range(0, 15)]
    # teachers_name.extend(
    #    ["model_A", "model_B", "model_C", "model_D", "model_E", "model_F", "model_G", "model_H", "model_I", "model_J",
    #     "model_K"])
    ori_malware_feature_vectors = read_pickle(config.get('feature_preprocess.drebin', 'malware_sample_vectors'))
    ori_malware_feature_labels = read_pickle(config.get('feature_preprocess.drebin', 'malware_sample_labels'))
    ori_benware_feature_vectors = read_pickle(config.get('feature_preprocess.drebin', 'benware_sample_vectors'))
    ori_benware_feature_labels = read_pickle(config.get('feature_preprocess.drebin', 'benware_sample_labels'))
    if ori_malware_feature_vectors.shape[0] > ori_benware_feature_vectors.shape[0]:
        ori_malware_feature_vectors, _, ori_malware_feature_labels, _ = \
            train_test_split(ori_malware_feature_vectors,
                             ori_malware_feature_labels,
                             train_size=ori_benware_feature_labels.shape[0])
    else:
        ori_benware_feature_vectors, _, ori_benware_feature_labels, _ = \
            train_test_split(ori_benware_feature_vectors,
                             ori_benware_feature_labels,
                             train_size=ori_malware_feature_labels.shape[0])

    ori_malware_feature_labels = to_categorical(ori_malware_feature_labels, num_classes=2)
    ori_benware_feature_labels = to_categorical(ori_benware_feature_labels, num_classes=2)

    all_feature_vectors = np.concatenate((ori_malware_feature_vectors, ori_benware_feature_vectors), axis=0)
    all_feature_labels = np.concatenate((ori_malware_feature_labels, ori_benware_feature_labels), axis=0)

    all_feature_vectors, all_feature_labels = sklearn.utils.shuffle(all_feature_vectors, all_feature_labels)
    ########################
    advtraining_method = "pgdl2"  # option: "pgdl2 pgdl1 pgdlinfinity jsma
    exp_name = '160x3'
    advtraining_number = 3  # 对于pgdl2来说，经历过一次adv_training就够了。
    # for adv_training_number in range(adv_training_total_numbers):
    teacher_models_graph = []
    teacher_models_sess = []
    teacher_models = []
    teacher_info = {}
    for j, teacher_name in enumerate(teachers_name):
        graph_j = tf.Graph()
        sess_j = tf.Session(graph=graph_j)
        with graph_j.as_default():
            with sess_j.as_default():
                parm = model_Param_dict[teacher_name]
                t = teacher_model(hyper_parameter=parm,
                                  model_name=teacher_name,
                                  is_trainable=True)
                # sess_j.run(tf.global_variables_initializer())
                teacher_saver = tf.train.Saver()
                teacher_model_load_dir = config.get('advtraining.drebin', 'advtraining_drebin_root') + \
                                         "/" + advtraining_method + "/" + teacher_name + "/adv" + str(
                    advtraining_number)

                print("teacher_model {} load parameter from {}".format(teacher_name, teacher_model_load_dir))
                t.load_param(teacher_model_load_dir, sess_j, teacher_saver)
                teacher_models.append(t)
        teacher_models_graph.append(graph_j)
        teacher_models_sess.append(sess_j)
    teacher_info['sess'] = teacher_models_sess
    teacher_info['graph'] = teacher_models_graph
    teacher_info['models'] = teacher_models
    ########################
    simulator_graph = tf.Graph()
    simulator_sess = tf.Session(graph=simulator_graph)
    with simulator_graph.as_default():
        with simulator_sess.as_default():
            simulator_model = MetaModel(advtraining_number=advtraining_number,
                                        advtraining_method=advtraining_method,
                                        distillation_method="pgdl2",
                                        exp_name=exp_name)
            simulator_saver = tf.train.Saver()
            simulator_sess.run(tf.global_variables_initializer())
            train_x, valid_x, train_y, valid_y = train_test_split(all_feature_vectors, all_feature_labels,
                                                                  test_size=0.2,
                                                                  random_state=np.random.randint(0, 999))
            simulator_model.warmup_training(train_x=train_x, train_y=train_y,
                                            valid_x=valid_x, valid_y=valid_y, sess=simulator_sess)
            simulator_model.meta_training(teacher_info,
                                          all_feature_vectors,
                                          all_feature_labels,
                                          simulator_sess,
                                          simulator_saver)


if __name__ == "__main__":
    _main()
