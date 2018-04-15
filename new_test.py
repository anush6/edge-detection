import os
import sys
import argparse
import yaml
#from urllib import parse as urlparse
import urllib.parse as urlparse
#from urllib.parse import urlparse
import urllib
from io import StringIO
import numpy as np
from PIL import Image
import tensorflow as tf
from hed.utils.io import IO
from hed.models.vgg16 import Vgg16

class Hed_test():
    def start(self,args):
        io = IO()
        self.cfgs = io.read_yaml_file(args.config_file)
        self.model = Vgg16(self.cfgs, run='testing')
        meta_model_file = "/Users/csrproject/edge/holy-edge/hed/models/hed-model-5001"
        saver = tf.train.Saver()
        session = self.get_session(args.gpu_limit)
        saver.restore(session, meta_model_file)
        self.model.setup_testing(session)
        im = self.fetch_img()
        edgemap = session.run(self.model.predictions, feed_dict={self.model.images: [im]})
        self.save_egdemaps(edgemap)


    def fetch_img(self):

        image = None
        image = Image.open("/Users/csrproject/edge/holy-edge/hed/data/HED-BSDS/test/india.jpg")
        image = image.resize((self.cfgs['testing']['image_width'], self.cfgs['testing']['image_height']))
        image = np.array(image, np.float32)
        image = self.colorize(image)
        image = image[:, :, self.cfgs['channel_swap']]
        image -= self.cfgs['mean_pixel_value']
        return image


    def colorize(self, image):

        if image.ndim == 2:
            image = image[:, :, np.newaxis]
            image = np.tile(image, (1, 1, 3))
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        return image

    def get_session(self,gpu_fraction):


        num_threads = int(os.environ.get('OMP_NUM_THREADS'))
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

        if num_threads:
            return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
        else:
            return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def save_egdemaps(self, em_maps):

        # Take the edge map from the network from side layers and fuse layer
        em_maps = [e[0] for e in em_maps]
        em_maps = em_maps + [np.mean(np.array(em_maps), axis=0)]
        count = 1
        for idx, em in enumerate(em_maps):


            em[em < self.cfgs['testing_threshold']] = 0.0

            em = 255.0 * (1.0 - em)
            em = np.tile(em, [1, 1, 3])

            em = Image.fromarray(np.uint8(em))
            if(count == 1):
                em.save('/Users/csrproject/Desktop/new'+str(count)+'.png')
            count+=1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Utility for Training/Testing DL models(Concepts/Captions) using theano/keras')
    parser.add_argument('--config-file', dest='config_file', type=str, help='Experiment configuration file')
    parser.add_argument('--gpu-limit', dest='gpu_limit', type=float, default=1.0, help='Use fraction of GPU memory (Useful with TensorFlow backend)')
    args = parser.parse_args()
    a = Hed_test()
    a.start(args)
