from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

class FCN8VGG:
  def __init__(self):
    self.model_path = './logs/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

  def build(self, num_classes=10, height=224, width=224, isTrain=True):
    image_input = Input(shape=(height, width, 3))

    # Layer 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(image_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Layer2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Layer3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)
    pool3 = x

    # Layer4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)
    pool4 = x

    # Layer5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)
    pool5 = x

    x = Flatten(name='fatten')(x)
    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dense(1000, activation='softmax', name='predictions')(x)

    vgg = Model(image_input, x)
    vgg.load_weights(self.model_path)

    conv_pool5 = Conv2D(4096, (7, 7), activation='relu', padding='same')(pool5)
    if isTrain:
        conv_pool5 = Dropout(0.5)(conv_pool5)
    conv_pool5 = Conv2D(4096, (1, 1), activation='relu', padding='same')(conv_pool5)
    if isTrain:
        conv_pool5 = Dropout(0.5)(conv_pool5)

    conv_pool5 = Conv2D(num_classes, (1, 1), kernel_initializer='he_normal')(conv_pool5)
    conv_pool5 = Conv2DTranspose(num_classes, kernel_size=(8, 8), strides=(4, 4), padding='same', use_bias=False)(conv_pool5)
    print('conv_pool5', conv_pool5)

    conv_pool4 = Conv2D(num_classes, (1, 1), kernel_initializer='he_normal')(pool4)
    conv_pool4 = Conv2DTranspose(num_classes, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False)(conv_pool4)
    print('conv_pool4', conv_pool4)

    conv_pool3 = Conv2D(num_classes, (1, 1), kernel_initializer='he_normal')(pool3)
    print('conv_pool3', conv_pool3)

    conv_x = Add()([conv_pool3, conv_pool4, conv_pool5])

    conv_x = Conv2DTranspose(num_classes, kernel_size=(16, 16), strides=(8, 8), padding='same', use_bias=False)(conv_x)
    print('conv_x', conv_x)
    out = Reshape((-1, num_classes))(conv_x)
    print('out', out)
    out = Activation('softmax')(out)
    model = Model(image_input, out)

    return model

