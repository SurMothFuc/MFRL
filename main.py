import os
import model
import tensorflow as tf



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定第一块GPU可用
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # 程序按需申请内存
    with tf.compat.v1.Session(config=config) as session:
        # with tf.compat.v1.Session() as session:
        sr = model.SRmodel(session, './dataset.h5', 200, 64, 2e-4, './testdataset.h5', './log',True)
        # sr.load_model('./model/model_data.ckpt')
        # sr.save_model('./model/model_data.ckpt')
        # sr.enlarge_YCrCb('./SelfExSR-data/Urban100/image_SRF_4_LR/img_019_SRF_4_LR.png', './output',scale=4)
        # sr.enlarge_YCrCb('./img_003_SRF_4_LR.png',  scale=4)
