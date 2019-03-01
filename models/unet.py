from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

class UNET:
  def build(self, num_classes=10, height=512, width=512, isTrain=True):
    image_input = Input(shape=(height, width, 3))

    #################################Encoder##############################
    # block1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(image_input)
    x = BatchNormalization()(x, training=isTrain)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = BatchNormalization()(x, training=isTrain)
    block1 = x

    # block2
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = BatchNormalization()(x, training=isTrain)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = BatchNormalization()(x, training=isTrain)
    block2 = x

    # block3
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = BatchNormalization()(x, training=isTrain)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = BatchNormalization()(x, training=isTrain)
    block3 = x

    # block4
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = BatchNormalization()(x, training=isTrain)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = BatchNormalization()(x, training=isTrain)
    block4 = x

    # block5
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = BatchNormalization()(x, training=isTrain)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = BatchNormalization()(x, training=isTrain)

    #################################Decoder##############################
    # block6
    x = Conv2DTranspose(512, kernel_size=(4, 4), strides=(2, 2), padding='same', name='deconv6_1')(x)
    x = BatchNormalization()(x, training=isTrain)
    x = Concatenate(axis=-1)([x, block4])
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv6_2')(x)
    x = BatchNormalization()(x, training=isTrain)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv6_3')(x)
    x = BatchNormalization()(x, training=isTrain)

    # block7
    x = Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding='same', name='deconv7_1')(x)
    x = BatchNormalization()(x, training=isTrain)
    x = Concatenate(axis=-1)([x, block3])
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv7_2')(x)
    x = BatchNormalization()(x, training=isTrain)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv7_3')(x)
    x = BatchNormalization()(x, training=isTrain)

    # block8
    x = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same', name='deconv8_1')(x)
    x = BatchNormalization()(x, training=isTrain)
    x = Concatenate(axis=-1)([x, block2])
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv8_2')(x)
    x = BatchNormalization()(x, training=isTrain)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv8_3')(x)
    x = BatchNormalization()(x, training=isTrain)

    # block9
    x = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='same', name='deconv9_1')(x)
    x = BatchNormalization()(x, training=isTrain)
    x = Concatenate(axis=-1)([x, block1])
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv9_2')(x)
    x = BatchNormalization()(x, training=isTrain)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv9_3')(x)
    x = BatchNormalization()(x, training=isTrain)
    x = Conv2D(num_classes, (1, 1), activation='relu', padding='same', name='conv4_3')(x)
    x = BatchNormalization()(x, training=isTrain)

    out = Reshape((-1, num_classes))(x)
    out = Activation('softmax')(out)
    model = Model(image_input, out)
    print(model.summary())

    return model
