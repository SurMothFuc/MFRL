import tensorflow as tf
import h5py
import numpy as np
import cv2
from matplotlib import pyplot as plt

import utils

import progressbar


class SRmodel(object):
    def __init__(self, session, dataset_path, epoch, batch_size, learning_rate, testdataset_path, log_path, istrain):
        self.session = session
        self.epoch = epoch
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.testdataset_path = testdataset_path
        self.build_graph_MFCNN_YCR_CB()
        self.merged = tf.compat.v1.summary.merge([self.loss_summary])
        self.test_merged_loss = tf.compat.v1.summary.merge([self.test_loss_summary])
        self.test_merged_psnr = tf.compat.v1.summary.merge([self.test_psnr_summary])
        if istrain:
            self.writer = tf.compat.v1.summary.FileWriter(log_path, self.session.graph)
        self.session.run(tf.compat.v1.global_variables_initializer())


    def build_graph_SRCNN_YCR_CB(self):
        self.data = tf.compat.v1.placeholder(tf.float32, [None, None, None, 1], name='images')
        self.label = tf.compat.v1.placeholder(tf.float32, [None, None, None, 1], name='labels')
        self.tep = tf.compat.v1.placeholder(tf.float32, [1], name='tep')
        self.test_loss_summary = tf.compat.v1.summary.scalar('test_loss', self.tep[0])  # 可视化观看常量
        self.test_psnr_summary = tf.compat.v1.summary.scalar('test_psnr', self.tep[0])  # 可视化观看常量

        Y = self.data

        self.w1_Y = tf.Variable(tf.random.normal([9, 9, 1, 64], stddev=1e-3), name='w1_Y')
        self.w2_Y = tf.Variable(tf.random.normal([1, 1, 64, 32], stddev=1e-3), name='w2_Y')
        self.w3_Y = tf.Variable(tf.random.normal([5, 5, 32, 1], stddev=1e-3), name='w3_Y')


        self.b1_Y = tf.Variable(tf.zeros([64]), name='b1_Y')
        self.b2_Y = tf.Variable(tf.zeros([32]), name='b2_Y')
        self.b3_Y = tf.Variable(tf.zeros([1]), name='b3_Y')

        self.conv1_Y = tf.nn.relu(self.conv2d(Y, self.w1_Y) + self.b1_Y)
        self.conv2_Y = tf.nn.relu(self.conv2d(self.conv1_Y, self.w2_Y) + self.b2_Y)
        self.conv3_Y = self.conv2d(self.conv2_Y, self.w3_Y) + self.b3_Y

        self.output = self.conv3_Y

        self.loss = tf.reduce_mean(tf.square(self.label - self.conv3_Y))
        self.loss_summary = tf.compat.v1.summary.scalar('loss', self.loss)  # 可视化观看常量

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        self.train = self.optimizer.minimize(self.loss)
        self.session.run(tf.compat.v1.global_variables_initializer())

    def build_graph_MFCNN_YCR_CB(self):  # 使用全局残差
        with tf.name_scope('inputs'):  # 结构化
            self.data = tf.compat.v1.placeholder(tf.float32, [None, None, None, 1], name='images')
            self.label = tf.compat.v1.placeholder(tf.float32, [None, None, None, 1], name='labels')
            self.tep = tf.compat.v1.placeholder(tf.float32, [1], name='tep')
            self.test_loss_summary = tf.compat.v1.summary.scalar('test_loss', self.tep[0])  # 可视化观看常量
            self.test_psnr_summary = tf.compat.v1.summary.scalar('test_psnr', self.tep[0])  # 可视化观看常量

        Y = self.data
        with tf.name_scope("Feature_extraction"):
            with tf.name_scope("convolution_3X3_3X3X1"):
                with tf.name_scope("weights"):
                    self.w1_Y_3 = tf.Variable(tf.random.normal([3, 3, 1, 16], stddev=1e-3), name='w1_Y_3')
                with tf.name_scope("biases"):
                    self.b1_Y_3 = tf.Variable(tf.zeros([16]), name='b1_Y_3')
                with tf.name_scope("convolution_relu"):
                    self.conv1_Y_3 = tf.nn.relu(self.conv2d(Y, self.w1_Y_3) + self.b1_Y_3)

            with tf.name_scope("convolution_5X5_3X3X2"):
                with tf.name_scope("convolution_5X5_3X3X2_1"):
                    with tf.name_scope("weights"):
                        self.w1_Y_5_1 = tf.Variable(tf.random.normal([3, 3, 1, 32], stddev=1e-3), name='w1_Y_5_1')
                    with tf.name_scope("biases"):
                        self.b1_Y_5_1 = tf.Variable(tf.zeros([32]), name='b1_Y_5_1')
                    with tf.name_scope("convolution_relu"):
                        self.conv1_Y_5_1 = tf.nn.relu(self.conv2d(Y, self.w1_Y_5_1) + self.b1_Y_5_1)
                with tf.name_scope("convolution_5X5_3X3X2_2"):
                    with tf.name_scope("weights"):
                        self.w1_Y_5_2 = tf.Variable(tf.random.normal([3, 3, 32, 32], stddev=1e-3), name='w1_Y_5_2')
                    with tf.name_scope("biases"):
                        self.b1_Y_5_2 = tf.Variable(tf.zeros([32]), name='b1_Y_5_2')
                    with tf.name_scope("convolution_relu"):
                        self.conv1_Y_5_2 = tf.nn.relu(self.conv2d(self.conv1_Y_5_1, self.w1_Y_5_2) + self.b1_Y_5_2)
            with tf.name_scope("convolution_7X7_3X3X3"):
                with tf.name_scope("convolution_7X7__3X3X3_1"):
                    with tf.name_scope("weights"):
                        self.w1_Y_7_1 = tf.Variable(tf.random.normal([3, 3, 1, 64], stddev=1e-3), name='w1_Y_7_1')
                    with tf.name_scope("biases"):
                        self.b1_Y_7_1 = tf.Variable(tf.zeros([64]), name='b1_Y_7_1')
                    with tf.name_scope("convolution_relu"):
                        self.conv1_Y_7_1 = tf.nn.relu(self.conv2d(Y, self.w1_Y_7_1) + self.b1_Y_7_1)
                with tf.name_scope("convolution_7X7_3X3X3_2"):
                    with tf.name_scope("weights"):
                        self.w1_Y_7_2 = tf.Variable(tf.random.normal([3, 3, 64, 64], stddev=1e-3), name='w1_Y_7_2')
                    with tf.name_scope("biases"):
                        self.b1_Y_7_2 = tf.Variable(tf.zeros([64]), name='b1_Y_7_2')
                    with tf.name_scope("convolution_relu"):
                        self.conv1_Y_7_2 = tf.nn.relu(self.conv2d(self.conv1_Y_7_1, self.w1_Y_7_2) + self.b1_Y_7_2)

                with tf.name_scope("convolution_7X7_3X3X3_3"):
                    with tf.name_scope("weights"):
                        self.w1_Y_7_3 = tf.Variable(tf.random.normal([3, 3, 64, 64], stddev=1e-3), name='w1_Y_7_3')
                    with tf.name_scope("biases"):
                        self.b1_Y_7_3 = tf.Variable(tf.zeros([64]), name='b1_Y_7_3')
                    with tf.name_scope("convolution_relu"):
                        self.conv1_Y_7_3 = tf.nn.relu(self.conv2d(self.conv1_Y_7_2, self.w1_Y_7_3) + self.b1_Y_7_3)
            with tf.name_scope("Concat"):
                with tf.name_scope("weights"):
                    self.w1_Y_CON = tf.Variable(tf.random.normal([1, 1, 112, 64], stddev=1e-3), name='w1_Y_CON')
                with tf.name_scope("biases"):
                    self.b1_Y_CON = tf.Variable(tf.zeros([64]), name='b1_Y_Con')
                with tf.name_scope("convolution_relu"):
                    self.conv1_Y_OUTCNN = tf.concat((self.conv1_Y_3, self.conv1_Y_5_2, self.conv1_Y_7_3), axis=3)
                    self.conv1_Y = tf.nn.relu(self.conv2d(self.conv1_Y_OUTCNN, self.w1_Y_CON) + self.b1_Y_CON)

        with tf.name_scope("Residual_layer"):
            self.conv2_res_Y_1 = self.res_block(self.conv1_Y, 64, 1)
            self.conv2_res_Y_2 = self.res_block(self.conv2_res_Y_1, 64, 2)
            self.conv2_res_Y_3 = self.res_block(self.conv2_res_Y_2, 64, 3)
            self.conv2_res_Y_4 = self.res_block(self.conv2_res_Y_3, 64, 4)
            self.conv2_res_Y_5 = self.res_block(self.conv2_res_Y_4, 64, 5)
            self.conv2_res_Y_6 = self.res_block(self.conv2_res_Y_5, 64, 6)
        with tf.name_scope("reconstruction"):
            with tf.name_scope("weights"):
                self.w3_Y = tf.Variable(tf.random.normal([5, 5, 6 * 64, 1], stddev=1e-3), name='w3_Y')
            with tf.name_scope("biases"):
                self.b3_Y = tf.Variable(tf.zeros([1]), name='b3_Y')
            with tf.name_scope("Feature_fusion_relu"):
                self.conv3concat = tf.concat(
                    (self.conv2_res_Y_1, self.conv2_res_Y_2, self.conv2_res_Y_3, self.conv2_res_Y_4,
                     self.conv2_res_Y_5, self.conv2_res_Y_6), axis=3)
                self.conv3_Y = self.conv2d(self.conv3concat, self.w3_Y) + self.b3_Y

        with tf.name_scope("Plus"):
            self.output = self.conv3_Y + Y
        with tf.name_scope("Loss"):
            self.loss = tf.reduce_mean(tf.square(self.label - self.output))
            self.loss_summary = tf.compat.v1.summary.scalar('loss', self.loss)  # 可视化观看常量
        with tf.name_scope('Train'):
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
            self.train = self.optimizer.minimize(self.loss)

    # 残差块
    def res_block(self, input_filter, input_filter_num, res_mark):
        with tf.name_scope(str.format("Res_block_{}", res_mark)):
            with tf.name_scope("Long_path_1"):
                with tf.name_scope("weights"):
                    wres_Y_1 = tf.Variable(tf.random.normal([3, 3, input_filter_num, 64], stddev=1e-3),
                                           name=str.format('wres_Y_{}_1', res_mark))
                with tf.name_scope("biases"):
                    bres_1 = tf.Variable(tf.zeros([64]), name=str.format('bres_{}_1', res_mark))
                with tf.name_scope("convolution"):
                    conv_res_1 = self.conv2d(input_filter, wres_Y_1) + bres_1
                with tf.name_scope("relu"):
                    relu1 = tf.nn.relu(conv_res_1)
            with tf.name_scope("Long_path_2"):
                with tf.name_scope("weights"):
                    wres_Y_2 = tf.Variable(tf.random.normal([3, 3, 64, 64], stddev=1e-3),
                                           name=str.format('wres_Y_{}_2', res_mark))
                with tf.name_scope("biases"):
                    bres_2 = tf.Variable(tf.zeros([64]), name=str.format('bres_{}_2', res_mark))
                with tf.name_scope("convolution"):
                    conv_res_2 = self.conv2d(relu1, wres_Y_2) + bres_2
                with tf.name_scope("relu"):
                    relu2 = tf.nn.relu(conv_res_2)
            with tf.name_scope("Broken_path"):
                with tf.name_scope("weights"):
                    wres_Y_1_L = tf.Variable(tf.random.normal([3, 3, input_filter_num, 64], stddev=1e-3),
                                             name=str.format('wres_Y_{}_1_L', res_mark))
                with tf.name_scope("biases"):
                    bres_1_L = tf.Variable(tf.zeros([64]), name=str.format('bres_{}_1_L', res_mark))
                with tf.name_scope("convolution"):
                    conv_res_1_L = self.conv2d(input_filter, wres_Y_1_L) + bres_1_L
                with tf.name_scope("relu"):
                    relu1_L = tf.nn.relu(conv_res_1_L)
            with tf.name_scope("Concat"):
                with tf.name_scope("weights"):
                    wres_cat = tf.Variable(tf.random.normal([1, 1, 128, input_filter_num], stddev=1e-3),
                                           name=str.format('wres_cat_{}', res_mark))
                with tf.name_scope("biases"):
                    bres_cat = tf.Variable(tf.zeros([input_filter_num]), name=str.format('bres_cat_{}', res_mark))
                with tf.name_scope("convolution"):
                    concat = tf.concat((relu2, relu1_L), axis=3)
                    out = self.conv2d(concat, wres_cat) + bres_cat
                with tf.name_scope("Plus"):
                    return out + input_filter

    def conv2d(self, input, filter):
        return tf.nn.conv2d(input=input, filters=filter, strides=[1, 1, 1, 1], padding="SAME")

    def train_model(self):
        psnr, mse = 0.0, 0.0
        data, label = self.read_data(self.dataset_path)
        testdata, testlabel = self.read_test(self.testdataset_path)
        widget = [
            ' [', progressbar.Timer(), '] ',
            progressbar.Percentage(),
            ' (', progressbar.SimpleProgress(), ') ',
            ' (', progressbar.ETA(), ') ', progressbar.FormatLabel(str.format('psnr:{:.3f} mse:{:.5f}', psnr, mse)),
        ]
        bar = progressbar.ProgressBar(widgets=widget, maxval=self.epoch * len(data) // self.batch_size)
        bar.start()
        for i in range(self.epoch):
            batch_data, batch_label = self.get_batchDataAndLabel(data, label, self.batch_size)
            for j in range(len(batch_data)):
                one_batch_data, one_batch_label = batch_data[j], batch_label[j]
                self.session.run(self.train, feed_dict={self.data: one_batch_data, self.label: one_batch_label})
                # print('epoch:', str(i), 'batch:', str(j), "process:", str.format("{:.3f}", j / len(batch_data) * 100),
                #       "%")
                bar.update(i * len(data) // self.batch_size + j)
                if j % 50 == 0 or j + 1 == len(batch_data):
                    result = self.session.run(self.merged, feed_dict={self.data: one_batch_data,
                                                                      self.label: one_batch_label})  # merged也是需要run的  
                    self.writer.add_summary(result, i * len(batch_data) + j)
                    psnr, mse = self.testPSNR(testdata, testlabel)
                    widget[10] = progressbar.FormatLabel(str.format('psnr:{:.3f} mse:{:.5f}', psnr, mse))
                    # print('psnr:',psnr,'mse:',mse)
                    result_mse = self.session.run(self.test_merged_loss, feed_dict={self.tep: np.array([mse])})
                    self.writer.add_summary(result_mse, i * len(batch_data) + j)
                    result_psnr = self.session.run(self.test_merged_psnr, feed_dict={self.tep: np.array([psnr])})
                    self.writer.add_summary(result_psnr, i * len(batch_data) + j)
        bar.finish()

    def save_model(self, savePath):
        saver = tf.compat.v1.train.Saver()
        saver.save(self.session, savePath)
        print("model saved")

    def load_model(self, savePath):
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.session, savePath)
        print('model loaded.')

    def read_data(self, path):
        with h5py.File(path, 'r') as file:
            try:
                data = np.array(file['train_data'])
                label = np.array(file['train_label'])
                return data, label
            except:
                print('load data failed!')

    def read_test(self, path):
        with h5py.File(path, 'r') as file:
            data = []
            label = []
            try:
                length = int(np.array(file['len']))
                for i in range(length):
                    data.append(np.array(file['test_data_' + str(i)]))
                    label.append(np.array(file['test_label_' + str(i)]))
                return data, label
            except:
                print('load data failed!')

    def get_batchDataAndLabel(self, data, label, batch_num):
        total = len(data)
        items = np.array(np.arange(0, total))
        np.random.shuffle(items)
        batch_data, batch_label, one_batch_data, one_batch_label = [], [], [], []
        for i in range(total):
            n = items[i]
            one_batch_data.append(data[n] / 255)
            one_batch_label.append(label[n] / 255)
            if (i + 1) % batch_num == 0 or i + 1 == total:
                batch_data.append(np.array(one_batch_data))
                batch_label.append(np.array(one_batch_label))
                one_batch_data, one_batch_label = [], []

        return np.array(batch_data), np.array(batch_label)

    def testPSNR(self, testdata, testlabel):
        psnrsum = []
        msesum = []
        for i in range(len(testdata)):
            Limage = testdata[i]
            Himage = testlabel[i]
            height, width, channel = Himage.shape
            Limage = cv2.cvtColor(Limage, cv2.COLOR_BGR2YCrCb)
            Limage = cv2.resize(Limage, (width, height), interpolation=cv2.INTER_CUBIC)
            net_input = np.reshape((Limage / 255)[:, :, 0], (1, height, width, 1))

            Cr = np.reshape((Limage / 255)[:, :, 1], (1, height, width, 1))
            Cb = np.reshape((Limage / 255)[:, :, 2], (1, height, width, 1))
            Y = self.session.run(self.output, feed_dict={self.data: net_input})
            B = (Cb - 0.5) / 0.564 + Y
            R = (Cr - 0.5) / 0.713 + Y
            G = (Y - 0.299 * R - 0.114 * B) / 0.587
            new_image = np.concatenate((B, G, R), axis=3)
            new_image = new_image * 255
            new_image = np.reshape(new_image, (height, width, channel))
            psnr, mse = utils.psnr(new_image, Himage)
            psnrsum.append(psnr)
            msesum.append(mse / 255 / 255)

        return np.mean(psnrsum), np.mean(msesum)

    def enlarge_YCrCb(self, low_image_path, scale=1):
        image = cv2.imread(low_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        if scale != 1:
            height, width, channel = image.shape
            image = cv2.resize(image, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_CUBIC)

        height, width, channel = image.shape
        net_input = np.reshape((image / 255)[:, :, 0], (1, height, width, 1))

        Cr = np.reshape((image / 255)[:, :, 1], (1, height, width, 1))
        Cb = np.reshape((image / 255)[:, :, 2], (1, height, width, 1))
        Y = self.session.run(self.output, feed_dict={self.data: net_input})

        B = (Cb - 0.5) / 0.564 + Y
        R = (Cr - 0.5) / 0.713 + Y
        G = (Y - 0.299 * R - 0.114 * B) / 0.587

        new_image = np.concatenate((B, G, R), axis=3)

        new_image = new_image * 255
        new_image = np.reshape(new_image, (height, width, channel))
        new_image = np.maximum(new_image, 0.0)
        new_image = np.minimum(new_image, 255.0)  # 数据规范汉化
        new_image = new_image.astype(np.uint8)

        return new_image
