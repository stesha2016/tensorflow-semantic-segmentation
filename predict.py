import tensorflow as tf
import cv2
import glob
import numpy as np
import itertools
import model_factory as mf

tf.app.flags.DEFINE_string('model_name', 'fcn16vgg', 'model name')
tf.app.flags.DEFINE_integer('size', 224, 'input height, width')
tf.app.flags.DEFINE_string('logs_path', './logs/boxes.h5', 'checkpoint files')
tf.app.flags.DEFINE_integer('num_classes', 2, '')
tf.app.flags.DEFINE_string('input_path', './data/boxes/test/', 'test files path')
tf.app.flags.DEFINE_string('output_path', './data/boxes/predict/', 'predict files path')
FLAGS = tf.app.flags.FLAGS

def getImage(file_path, size):
  img = cv2.imread(file_path)
  h, w = img.shape[0], img.shape[1]
  img_resize = cv2.resize(img, (size, size))
  img_resize = img_resize.astype(np.float32)
  img_resize[:, :, 0] -= 103.939
  img_resize[:, :, 1] -= 116.779
  img_resize[:, :, 2] -= 123.68
  return img_resize, img, h, w

def get_rect(file_path):
  image = cv2.imread(file_path)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  ele = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
  minus_mor = cv2.dilate(gray, ele)
  print(minus_mor.shape)

  retval, labels, stats, centroids = cv2.connectedComponentsWithStats(minus_mor, connectivity=8)
  max_area = 0
  mark = 0
  for j in range(retval):
    if stats[j][cv2.CC_STAT_AREA] < image.shape[0] * image.shape[1] and stats[j][cv2.CC_STAT_AREA] > max_area:
      mark = j
      max_area = stats[j][cv2.CC_STAT_AREA]
      pos = stats[j].tolist()
      print(pos)
  return pos

def mark_rect(input_path, base_image):
  print('mark_rect {}'.format(input_path))
  pos = get_rect(input_path)
  image_output = cv2.rectangle(base_image, (pos[0], pos[1]), (pos[0] + pos[2], pos[1] + pos[3]), (255, 0, 0), 2)
  file_name = input_path.split('/')[-1].split('.')[0]
  output_path = input_path.replace(file_name, file_name + '_rect')
  cv2.imwrite(output_path, image_output)

def predict():
  if FLAGS.input_path[len(FLAGS.input_path) - 1] != '/':
    FLAGS.input_path = FLAGS.input_path + '/'
  if FLAGS.output_path[len(FLAGS.output_path) - 1] != '/':
    FLAGS.output_path = FLAGS.output_path + '/'

  myModel = mf.modelSet[FLAGS.model_name]()
  m = myModel.build(FLAGS.num_classes, FLAGS.size, FLAGS.size)
  m.load_weights(FLAGS.logs_path)
  m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  colors = [(0, 0, 0), (255, 255, 255)]

  images = glob.glob(FLAGS.input_path + '*.jpg') + glob.glob(FLAGS.input_path + '*.png') + glob.glob(FLAGS.input_path + '*.jpeg')
  for image in images:
    image_name = image.split('/')[-1]
    print('now predict {}'.format(image_name))
    output_path = FLAGS.output_path + image_name

    X, origin_image, h, w = getImage(image, FLAGS.size)
    X = X.reshape((1, X.shape[0], X.shape[1], X.shape[2]))
    predict = m.predict(X)[0]
    predict = predict.reshape((FLAGS.size, FLAGS.size, FLAGS.num_classes)).argmax(axis=2)
    seg_img = np.zeros((FLAGS.size, FLAGS.size, 3))

    for c in range(FLAGS.num_classes):
      seg_img[:, :, 0] = (predict[:, :] * colors[c][0]).astype('uint8')
      seg_img[:, :, 1] = (predict[:, :] * colors[c][1]).astype('uint8')
      seg_img[:, :, 2] = (predict[:, :] * colors[c][2]).astype('uint8')
    seg_img = cv2.resize(seg_img, (w, h))
    cv2.imwrite(output_path , seg_img)

    mark_rect(output_path, origin_image)

if __name__ == '__main__':
  predict()
