from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten, LeakyReLU
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, TimeDistributed, BatchNormalization
from keras import backend as K
from RoiPoolingConv import RoiPoolingConv
from FixedBatchNormalization import FixedBatchNormalization
import h5py


def darknet_base(input_tensor=None, trainable = False):

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

	x = ZeroPadding2D((3, 3))(img_input)

	x = Convolution2D(16, 3, 3, init='he_normal', activation='linear', border_mode='same', trainable=trainable)(x)
	x = BatchNormalization(axis=1, mode = 0)(x)
	x = LeakyReLU()(x)

	x = MaxPooling2D((2, 2))(x)

	x = Convolution2D(32, 3, 3, init='he_normal', activation='linear', border_mode='same', trainable=trainable)(x)
	x = BatchNormalization(axis=1, mode = 0)(x)
	x = LeakyReLU()(x)

	x = MaxPooling2D((2, 2))(x)

	x = Convolution2D(64, 3, 3, init='he_normal', activation='linear', border_mode='same', trainable=trainable)(x)
	x = BatchNormalization(axis=1, mode = 0)(x)
	x = LeakyReLU()(x)

	x = MaxPooling2D((2, 2))(x)

	x = Convolution2D(128, 3, 3, init='he_normal', activation='linear', border_mode='same', trainable=trainable)(x)
	x = BatchNormalization(axis=1, mode = 0)(x)
	x = LeakyReLU()(x)

	x = MaxPooling2D((2, 2))(x)

	x = Convolution2D(256, 3, 3, init='he_normal', activation='linear', border_mode='same', trainable=trainable)(x)
	x = BatchNormalization(axis=1, mode = 0)(x)
	x = LeakyReLU()(x)

	x = MaxPooling2D((2, 2))(x)

	x = Convolution2D(512, 3, 3, init='he_normal', activation='linear', border_mode='same', trainable=trainable)(x)
	x = BatchNormalization(axis=1, mode = 0)(x)
	x = LeakyReLU()(x)

	x = MaxPooling2D((2, 2),border_mode='same')(x)

	x = Convolution2D(1024, 3, 3, init='he_normal', activation='linear', border_mode='same', trainable=trainable)(x)
	x = BatchNormalization(axis=1, mode = 0)(x)
	x = LeakyReLU()(x)

	return x



def classifier_layers(x, nb_classes):
	x = TimeDistributed(Convolution2D(nb_classes, 1, 1, init='he_normal', activation='linear', border_mode='same', trainable=True), name = 'fc_{}'.format(nb_classes))(x)
	x = TimeDistributed(LeakyReLU())(x)
	x = TimeDistributed(AveragePooling2D((4,4)))(x)
	
	return x

def rpn(base_layers,num_anchors):

    x = Convolution2D(512, 3, 3, border_mode = 'same', activation='relu', init='normal',name='rpn_conv1')(base_layers)

    x_class = Convolution2D(num_anchors, 1, 1, activation='sigmoid', init='normal',name='rpn_out_class')(x)
    x_regr = Convolution2D(num_anchors * 4, 1, 1, activation='linear', init='normal',name='rpn_out_regr')(x)

    return [x_class,x_regr]

def classifier(base_layers,input_rois,num_rois,nb_classes = 21):

    pooling_regions = 7

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers,input_rois])
    out = classifier_layers(out_roi_pool, nb_classes)
    out = TimeDistributed(Flatten(),name='td_flatten')(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax'), name='dense_class_{}'.format(nb_classes))(out)
    out_regr = TimeDistributed(Dense(4, activation='linear'), name='dense_regr')(out)
    
    return [out_class,out_regr]