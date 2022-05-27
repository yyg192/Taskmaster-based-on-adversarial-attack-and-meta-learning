import numpy as np

class DataProducer(object):
    def __init__(self, dataX, datay, batch_size, n_epochs = None, n_steps=None, name='train'):
        '''
        The data factory yield data at designated batch size and steps
        :param dataX: 2-D array numpy type supported. shape: [num, feat_dims]
        :param datay: 2-D or 1-D array.
        :param batch_size: setting batch size for training or testing. Only integer supported.
        :param n_epochs: setting epoch for training. The default value is None
        :param n_steps: setting global steps for training. The default value is None. If provided, param n_epochs will be neglected.
        :param name: 'train' or 'test'. if the value is 'test', the n_epochs will be set to 1.
        '''
        try:
            assert(name=='train' or name == 'test' or name == 'val')
        except Exception as e:
            raise AssertionError("Only support selections: 'train', 'val' or 'test'.\n")

        self.dataX = dataX
        self.datay = datay
        self.batch_size = batch_size
        self.mini_batches = self.dataX.shape[0] // self.batch_size
        if self.dataX.shape[0] % self.batch_size > 0:
            self.mini_batches = self.mini_batches + 1
            if (self.dataX.shape[0] > self.batch_size) and \
                    (name == 'train' or name == 'val'):
                np.random.seed(0)
                rdm_idx = np.random.choice(self.dataX.shape[0], self.batch_size - self.dataX.shape[0] % self.batch_size, replace=False)
                self.dataX = np.vstack([dataX, dataX[rdm_idx]])
                self.datay = np.concatenate([datay, datay[rdm_idx]])

        if name == 'train':
            if n_epochs is not None:
                self.steps = n_epochs * self.mini_batches
            elif n_steps is not None:
                self.steps = n_steps
            else:
                self.steps = None
        if name == 'test' or name == 'val':
            self.steps = None

        self.name = name
        self.cursor = 0
        if self.steps is None:
            self.max_iterations = self.mini_batches
        else:
            self.max_iterations = self.steps

    def next_batch(self):
        while self.cursor < self.max_iterations:
            pos_cursor = self.cursor % self.mini_batches
            start_i = pos_cursor * self.batch_size

            end_i = start_i + self.batch_size
            if end_i > self.dataX.shape[0]:
                end_i = self.dataX.shape[0]
            if start_i == end_i:
                break

            yield self.cursor, self.dataX[start_i:end_i], self.datay[start_i: end_i]
            self.cursor = self.cursor + 1

    def next_batch2(self):
        while self.cursor < self.max_iterations:
            pos_cursor = self.cursor % self.mini_batches
            start_i = pos_cursor * self.batch_size
            if start_i == self.dataX.shape[0]:
                start_i = 0
            end_i = start_i + self.batch_size
            if end_i > self.dataX.shape[0]:
                end_i = self.dataX.shape[0]
            self.cursor = self.cursor + 1
            yield self.cursor, start_i, end_i, self.dataX[start_i:end_i], self.datay[start_i: end_i]

    def reset_cursor(self):
        self.cursor = 0

    def get_current_cursor(self):
        return self.cursor

class DataProducer2(object):
    def __init__(self, dataX, datay,soft_datay, batch_size, n_epochs = None, n_steps=None, name='train'):
        '''
        The data factory yield data at designated batch size and steps
        :param dataX: 2-D array numpy type supported. shape: [num, feat_dims]
        :param datay: 2-D or 1-D array.
        :param batch_size: setting batch size for training or testing. Only integer supported.
        :param n_epochs: setting epoch for training. The default value is None
        :param n_steps: setting global steps for training. The default value is None. If provided, param n_epochs will be neglected.
        :param name: 'train' or 'test'. if the value is 'test', the n_epochs will be set to 1.
        '''
        try:
            assert(name=='train' or name == 'test' or name == 'val')
        except Exception as e:
            raise AssertionError("Only support selections: 'train', 'val' or 'test'.\n")

        self.dataX = dataX
        self.datay = datay
        self.soft_datay = soft_datay
        self.batch_size = batch_size
        self.mini_batches = self.dataX.shape[0] // self.batch_size
        if self.dataX.shape[0] % self.batch_size > 0:
            self.mini_batches = self.mini_batches + 1
            if (self.dataX.shape[0] > self.batch_size) and \
                    (name == 'train' or name == 'val'):
                np.random.seed(0)
                rdm_idx = np.random.choice(self.dataX.shape[0], self.batch_size - self.dataX.shape[0] % self.batch_size, replace=False)
                self.dataX = np.vstack([dataX, dataX[rdm_idx]])
                self.datay = np.concatenate([datay, datay[rdm_idx]])
                self.soft_datay = np.concatenate([soft_datay, soft_datay[rdm_idx]])

        if name == 'train':
            if n_epochs is not None:
                self.steps = n_epochs * self.mini_batches
            elif n_steps is not None:
                self.steps = n_steps
            else:
                self.steps = None
        if name == 'test' or name == 'val':
            self.steps = None

        self.name = name
        self.cursor = 0
        if self.steps is None:
            self.max_iterations = self.mini_batches
        else:
            self.max_iterations = self.steps

    def next_batch(self):
        while self.cursor < self.max_iterations:
            pos_cursor = self.cursor % self.mini_batches
            start_i = pos_cursor * self.batch_size

            end_i = start_i + self.batch_size
            if end_i > self.dataX.shape[0]:
                end_i = self.dataX.shape[0]
            if start_i == end_i:
                break

            yield self.cursor, self.dataX[start_i:end_i], self.datay[start_i: end_i], self.soft_datay[start_i: end_i]
            self.cursor = self.cursor + 1

    def next_batch2(self):
        while self.cursor < self.max_iterations:
            pos_cursor = self.cursor % self.mini_batches
            start_i = pos_cursor * self.batch_size
            if start_i == self.dataX.shape[0]:
                start_i = 0
            end_i = start_i + self.batch_size
            if end_i > self.dataX.shape[0]:
                end_i = self.dataX.shape[0]
            self.cursor = self.cursor + 1
            yield self.cursor, start_i, end_i, self.dataX[start_i:end_i], self.datay[start_i: end_i], self.soft_datay[start_i: end_i]

    def reset_cursor(self):
        self.cursor = 0

    def get_current_cursor(self):
        return self.cursor