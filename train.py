import models
import tensorflow as tf
import cv2
import glob
import numpy as np
import itertools
import model_factory as mf

tf.app.flags.DEFINE_string('model_name', 'fcn8vgg', 'model name')
tf.app.flags.DEFINE_integer('size', 224, 'input height, width')
tf.app.flags.DEFINE_string('logs_path', './logs', 'checkpoint files')
tf.app.flags.DEFINE_integer('num_classes', 2, '')
tf.app.flags.DEFINE_integer('batch_size', 16, '')
tf.app.flags.DEFINE_integer('epochs', 20, '')
tf.app.flags.DEFINE_string('img_path', './data/boxes/train/', 'image path')
tf.app.flags.DEFINE_string('seg_path', './data/boxes/train_segmentation', 'segmentation path')
# tf.app.flags.DEFINE_boolean('clone_on_cpu', True,
#                             'Use CPUs to deploy clones.')
FLAGS = tf.app.flags.FLAGS

def getImage(file_path, size):
  img = cv2.imread(file_path)
  img = cv2.resize(img, (size, size))
  img = img.astype(np.float32)
  img[:, :, 0] -= 103.939
  img[:, :, 1] -= 116.779
  img[:, :, 2] -= 123.68
  return img

def getSeg(seg_path, size, num_classes):
  seg = cv2.imread(seg_path)
  seg = cv2.resize(seg, (size, size))[:, :, 0]

  labels = np.zeros((size, size, num_classes))
  for c in range(num_classes):
    labels[:, :, c] = (seg == c).astype(int)

  labels = labels.reshape(size * size, num_classes)
  return labels

def minibatch(img_path, seg_path, num_classes, batch_size, img_size):
  if img_path[len(img_path) - 1] != '/':
    img_path = img_path + '/'
  if seg_path[len(seg_path) - 1] != '/':
    seg_path = seg_path + '/'

  images = glob.glob(img_path + '*.jpg') + glob.glob(img_path + '*.png') + glob.glob(img_path + '*.jpeg')
  images.sort()
  segs = glob.glob(seg_path + '*.jpg') + glob.glob(seg_path + '*.png') + glob.glob(seg_path + '*.jpeg')
  segs.sort()

  assert len(images) == len(segs)
  print('training set number is {}'.format(len(images)))
  for im, seg in zip(images, segs):
    assert(im.split('/')[-1].split('.')[0] == seg.split('/')[-1].split('.')[0])

  iterator = itertools.cycle(zip(images, segs))

  while True:
    X = []
    Y = []
    for _ in range(batch_size):
      im, seg = iterator.next()
      X.append(getImage(im, img_size))
      Y.append(getSeg(seg, img_size, num_classes))
      yield np.array(X), np.array(Y)

def train():
  myModel = mf.modelSet[FLAGS.model_name]()
  m = myModel.build(FLAGS.num_classes, FLAGS.size, FLAGS.size)
  m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  batch = minibatch(FLAGS.img_path, FLAGS.seg_path, FLAGS.num_classes, FLAGS.batch_size, FLAGS.size)

  m.fit_generator(batch, FLAGS.batch_size, epochs=FLAGS.epochs, shuffle=False)
  m.save('./logs/boxes.h5')
  #m.save_weights('./logs/checkpoint')

if __name__ == '__main__':
  train()
