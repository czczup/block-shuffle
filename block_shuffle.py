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
parser.add_argument('--input', type=str, default=None, help="path of the content image")
parser.add_argument('--output', type=str, default=None, help="path to save the output image")
parser.add_argument('--model', type=str, default=None, help="path of the model")
parser.add_argument('--max-width', type=int, default=1000, help="max width of sub-images")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


class BlockItem:
    def __init__(self, id, block):
        self.id = id
        self.block = block


def unpadding(image, padding):
    width, height = image.shape[1], image.shape[0]
    image = image[padding:height - padding, padding:width - padding]
    return image


def expand(image):
    top = bottom = (height_expand - height) // 2
    left = right = (width_expand - width) // 2
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT)
    return new_image


def cut(image, block_size, padding, width, height):
    row_num = math.ceil(height / block_size)
    col_num = math.ceil(width / block_size)
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


def shuffle(block_list):
    object_list = [BlockItem(id, block) for id, block in enumerate(block_list)]
    np.random.shuffle(object_list)
    block_list = [item.block for item in object_list]
    return block_list, object_list


def concat(block_list, max_width):
    def list2subimage(image_list, block_size, max_size):
        num = max_size // block_size
        images = []
        for sublist in image_list:
            temp_list = [sublist[i:i + num] for i in range(0, len(sublist), num)]
            row_imgs = []
            for i, row in enumerate(temp_list):
                row_img = cv2.hconcat(row)
                row_imgs.append(row_img)
            new_image = cv2.vconcat(row_imgs)
            images.append(new_image)
        return images

    block_num = (max_width // block_width)**2
    block_list = [block_list[i:i+block_num] for i in range(0, len(block_list), block_num)]
    block_list[-1] += block_list[0][0:block_num-len(block_list[-1])]
    return list2subimage(block_list, block_width, max_width)


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
                image_output = sess.run(output_op, feed_dict={
                    input_op: [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]
                })[0]
                image_output = np.clip(image_output, 0, 255).astype(np.uint8)
                image_output = cv2.cvtColor(image_output, cv2.COLOR_RGB2BGR)
                result_list.append(image_output)
            sys.stdout.write("\n")
    return result_list


def recut(subimage_list, object_list):
    block_list = []
    for image in subimage_list:
        blocks = cut(image, block_width, padding=0, width=image.shape[1], height=image.shape[0])
        block_list += blocks
    block_list = block_list[0:total_block_num]
    for index, block in enumerate(block_list):
        object_list[index].image = block
    return object_list


def sort(object_list):
    object_list = sorted(object_list, key=lambda item: item.id)
    block_list = [item.image for item in object_list]
    return block_list


def restore(block_list):
    def horizontal_blend(overlap1, overlap2):
        overlap1 = np.array(overlap1)
        overlap2 = np.array(overlap2)
        target = (overlap1 * (1 - horizontal_alpha) + overlap2 * horizontal_alpha).astype(np.uint8)
        return target

    def vertical_blend(overlap1, overlap2):
        overlap1 = np.array(overlap1)
        overlap2 = np.array(overlap2)
        target = (overlap1 * (1 - vertical_alpha) + overlap2 * vertical_alpha).astype(np.uint8)
        return target

    block_list = [unpadding(block, padding=border_width) for block in block_list]
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

    start_x = (width_expand - width) // 2 - border_width
    start_y = (height_expand - height) // 2 - border_width
    restored_image = image[start_y:start_y + height, start_x:start_x + width]
    return restored_image


def smooth(image):
    for i in range(4):
        image = cv2.bilateralFilter(src=image, d=0, sigmaColor=10, sigmaSpace=10)
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


def block_shuffle(image, sess):
    global basic_width, padding_width
    global border_width, blending_width, width_expand, height_expand
    global width, height, block_width, total_block_num
    global horizontal_alpha, vertical_alpha

    """ prepare """
    basic_width = 16
    padding_width = 16

    block_width = basic_width+padding_width*2
    border_width = 8
    blending_width = padding_width-border_width
    width = image.shape[1]
    height = image.shape[0]
    width_expand = math.ceil(width/basic_width)*basic_width+2*padding_width
    height_expand = math.ceil(height/basic_width)*basic_width+2*padding_width
    total_block_num = math.ceil(width/basic_width)*math.ceil(height/basic_width)

    _ = np.array([[[i/(2*blending_width)] for i in range(2*blending_width)]])
    horizontal_alpha = np.tile(_, (block_width-2*border_width, 1, 3))
    _ = np.array([[[i/(2*blending_width)]] for i in range(2*blending_width)])
    vertical_alpha = np.tile(_, (1, width_expand-2*border_width, 3))

    """ start """
    time_start = time.time()
    image = expand(image)
    block_list = cut(image, basic_width, padding_width, width=width, height=height)
    block_list, object_list = shuffle(block_list)
    subimage_list = concat(block_list, max_width=args.max_width)
    preprocessing_time = time.time()-time_start
    # print("pre-processing time: %.2fs" % preprocessing_time)

    time_start = time.time()
    subimage_list = style_transfer(subimage_list, sess)
    style_transfer_time = time.time()-time_start
    # print("style transfer time: %.2fs" % style_transfer_time)

    time_start = time.time()
    object_list = recut(subimage_list, object_list)
    block_list = sort(object_list)
    image = restore(block_list)
    image = smooth(image)
    postprocessing_time = time.time()-time_start
    # print("post-processing time: %.2fs" % postprocessing_time)
    print("total time: %.2fs"%(preprocessing_time+style_transfer_time+postprocessing_time))

    return image


if __name__ == '__main__':
    sess = load_model(args.model)
    image = cv2.imread(args.input)
    image = block_shuffle(image, sess)
    cv2.imwrite(args.output, image)
