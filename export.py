import tensorflow as tf
import sys
sys.path.insert(0, 'src')
import transform
import os
import random
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--gpu', type=str, default='0', help="choose a GPU")
parser.add_argument('--input', type=str, default=None, help="the path of the checkpoint file")
parser.add_argument('--output', type=str, default=None, help="the path to save the .pb file")
parser.add_argument("--generate_cover_image", default='False', action='store_true', help='generate cover image or not')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def get_coco2014_dataset():
    path_dir = os.listdir('data/train2014')
    all_coco2014 = []
    for file in path_dir:
        all_coco2014.append('data/train2014/'+file)
    image_list = []
    for i in range(20):
        image_list.append(all_coco2014[random.randint(0, len(all_coco2014)-1)])
    return image_list


def center_crop(image, x, y):
    width, height = image.size[0], image.size[1]
    crop_side = min(width, height)
    width_crop = (width-crop_side)//2
    height_crop = (height-crop_side)//2
    box = (width_crop, height_crop, width_crop+crop_side, height_crop+crop_side)
    image = image.crop(box)
    image = image.resize((x, y), Image.ANTIALIAS)
    return image


def unpadding(image, padding):
    width, height = image.size
    box = (padding, padding, width-padding, height-padding)
    image = image.crop(box)
    return image


def export(ckpt_file, pb_file, generate_cover_image=False):
    g = tf.Graph()
    with g.as_default():
        with tf.Session() as sess:
            image_placeholder = tf.placeholder(tf.float32, [1, None, None, 3], name='input')
            generated_image = transform.net_v2(image_placeholder)
            clip_image = tf.clip_by_value(generated_image, 0, 255, name="output")
            cast_image = tf.cast(clip_image, tf.uint8)
            saver = tf.train.Saver(tf.global_variables())
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            saver.restore(sess, ckpt_file)

            if generate_cover_image == True:
                coco2014_list = get_coco2014_dataset()
                for idx, filename in enumerate(coco2014_list):
                    image = Image.open(filename)
                    image = image.convert('RGB')
                    image_input = center_crop(image, 960, 960)
                    image_output = sess.run(cast_image, feed_dict={
                        image_placeholder: [np.array(image_input)]
                    })[0]
                    image = Image.fromarray(image_output)
                    image = unpadding(image, padding=30)
                    image = center_crop(image, 512, 512)
                    image.save(ckpt_file.split("fns")[0]+filename.split("/")[-1])

            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, output_node_names=['output'])

            with tf.gfile.FastGFile(pb_file, mode='wb') as f:
                f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
    export(args.input, args.output, args.generate_cover_image)


