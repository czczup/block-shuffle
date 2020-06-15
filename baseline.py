import tensorflow as tf
import numpy as np
import os
import cv2
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--gpu', type=str, default='0', help="choose a GPU")
parser.add_argument('--input', type=str, default=None, help="the path of the content image")
parser.add_argument('--output', type=str, default=None, help="the path to save the output image")
parser.add_argument('--model', type=str, default=None, help="the path of the model")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def unpadding(image, padding):
    width, height = image.shape[1], image.shape[0]
    image = image[padding:height - padding, padding:width - padding]
    return image


def padding(image, padding):
    new_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT)
    return new_image


def style_transfer(image, sess):
    with sess.graph.as_default():
        with sess.as_default():
            input_op = sess.graph.get_tensor_by_name("input:0")
            output_op = sess.graph.get_tensor_by_name("output:0")
            image_output = sess.run(output_op, feed_dict={
                input_op: [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]
            })[0]
            image_output = np.clip(image_output, 0, 255).astype(np.uint8)
            image_output = cv2.cvtColor(image_output, cv2.COLOR_RGB2BGR)
    return image_output


def load_model(model_name):
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    with sess.graph.as_default():
        with sess.as_default():
            tf.global_variables_initializer().run()
            with tf.gfile.FastGFile(model_name, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
    return sess


def baseline(image, sess):
    image = padding(image, padding=40)
    image = style_transfer(image, sess)
    image = unpadding(image, padding=40)
    return image


if __name__ == '__main__':
    sess = load_model(args.model)
    image = cv2.imread(args.input)
    image = baseline(image, sess)
    cv2.imwrite(args.output, image)
