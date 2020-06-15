import tensorflow as tf
import numpy as np
import os
import sys
import cv2
import time
import math
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


def cut(image, block_size, padding, width, height):
    row_num = math.ceil(height / block_size)
    col_num = math.ceil(width / block_size)  # 行数和列数
    image_list = []
    for j in range(0, row_num):
        b = j * block_size
        d = (j + 1) * block_size + 2 * padding
        for i in range(0, col_num):
            a = i * block_size
            c = (i + 1) * block_size + 2 * padding
            image_block = image[b:d, a:c]
            image_list.append(image_block)
    return image_list


def style_transfer(subimage_list, sess):
    with sess.graph.as_default():
        with sess.as_default():
            input_op = sess.graph.get_tensor_by_name("input:0")
            output_op = sess.graph.get_tensor_by_name("output:0")
            result_list = []
            length = len(subimage_list)
            for index, image in enumerate(subimage_list):
                sys.stdout.write('\r>> Style transfer %d/%d' % (index + 1, length))
                sys.stdout.flush()
                image = padding(image, padding=20)
                image_output = sess.run(output_op, feed_dict={
                    input_op: [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]
                })[0]
                image_output = np.clip(image_output, 0, 255).astype(np.uint8)
                image_output = cv2.cvtColor(image_output, cv2.COLOR_RGB2BGR)
                image_output = unpadding(image_output, padding=20)
                result_list.append(image_output)
            sys.stdout.write("\n")
    return result_list


def restore(block_list):
    def horizontal_blend(overlap1, overlap2):
        overlap1 = np.array(overlap1)
        overlap2 = np.array(overlap2)
        _ = np.array([[[i/(2*blending_width)] for i in range(2*blending_width)]])
        horizontal_alpha = np.tile(_, (overlap1.shape[0], 1, 3))
        target = (overlap1 * (1 - horizontal_alpha) + overlap2 * horizontal_alpha).astype(np.uint8)
        return target

    def vertical_blend(overlap1, overlap2):
        overlap1 = np.array(overlap1)
        overlap2 = np.array(overlap2)
        _ = np.array([[[i/(2*blending_width)]] for i in range(2*blending_width)])
        vertical_alpha = np.tile(_, (1, width_expand, 3))
        target = (overlap1 * (1 - vertical_alpha) + overlap2 * vertical_alpha).astype(np.uint8)
        return target

    row_num = math.ceil(height / basic_width)
    col_num = math.ceil(width / basic_width)
    block_list = [block_list[i:i+col_num] for i in range(0, len(block_list), col_num)]
    row_images = []

    for i, row in enumerate(block_list):
        row_image = None
        for j, item in enumerate(row):
            if j == 0:
                row_image = item
            else:
                if blending_width != 0:
                    left = row_image[:, 0:-2*blending_width]
                    overlap1 = row_image[:, -2*blending_width:]
                    overlap2 = item[:, 0:2*blending_width]
                    right = item[:, 2*blending_width:]
                    overlap = horizontal_blend(overlap1, overlap2)
                    row_image = cv2.hconcat([left, overlap, right])
                else:
                    row_image = cv2.hconcat([row_image, item])
        row_images.append(row_image)

    image = None
    for i, row in enumerate(row_images):
        if i == 0:
            image = row
        else:
            if blending_width != 0:
                top = image[0:-2*blending_width, :]
                overlap1 = image[-2*blending_width:, :]
                overlap2 = row[0:2*blending_width, :]
                bottom = row[2*blending_width:, :]
                overlap = vertical_blend(overlap1, overlap2)
                image = cv2.vconcat([top, overlap, bottom])
            else:
                image = cv2.vconcat([image, row])

    return image


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


def feathering(image, sess):
    global basic_width, padding_width, blending_width
    global block_width, width, height, width_expand, height_expand

    """ prepare """
    basic_width = 1000
    padding_width = 100

    block_width = basic_width+padding_width*2
    blending_width = padding_width
    width = image.shape[1]
    height = image.shape[0]
    width_expand = width+2*padding_width
    height_expand = height+2*padding_width

    """ start """
    time_start = time.time()
    image = padding(image, padding_width)
    subimage_list = cut(image, basic_width, padding_width, width=width, height=height)
    subimage_list = style_transfer(subimage_list, sess)
    image = restore(subimage_list)
    image = unpadding(image, padding_width)
    print("total time: %.2fs"%(time.time()-time_start))
    return image


if __name__ == '__main__':
    sess = load_model(args.model)
    image = cv2.imread(args.input)
    image = feathering(image, sess)
    cv2.imwrite(args.output, image)
