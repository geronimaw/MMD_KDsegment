from tensorflow.keras.models import *
from tensorflow.keras.layers import *

def JointDetModelFull(first_layer_channels=64, joint_num=12, compo_num=8, input_size=(96,128,1)):
    inputs = Input(input_size)

    # Encoder
    # First Step
    conv1 = Conv2D(filters=first_layer_channels, kernel_size=3, strides=1, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    # Encoder 1
    branch_1_Enc_1 = Conv2D(filters=first_layer_channels, kernel_size=2, strides=2)(conv1)
    branch_1_Enc_1 = BatchNormalization()(branch_1_Enc_1)
    branch_1_Enc_1 = Activation('relu')(branch_1_Enc_1)

    branch_1_Enc_1 = Conv2D(filters=first_layer_channels, kernel_size=3, strides=1, padding='same')(branch_1_Enc_1)
    branch_1_Enc_1 = BatchNormalization()(branch_1_Enc_1)
    branch_1_Enc_1 = Activation('relu')(branch_1_Enc_1)

    branch_2_Enc_1 = Conv2D(filters=first_layer_channels, kernel_size=2, strides=2)(conv1)
    branch_2_Enc_1 = BatchNormalization()(branch_2_Enc_1)
    branch_2_Enc_1 = Activation('relu')(branch_2_Enc_1)

    branch_2_Enc_1 = Conv2D(filters=first_layer_channels, kernel_size=3, strides=1, padding='same')(branch_2_Enc_1)
    branch_2_Enc_1 = BatchNormalization()(branch_2_Enc_1)
    branch_2_Enc_1 = Activation('relu')(branch_2_Enc_1)

    Enc_1 = concatenate([branch_1_Enc_1, branch_2_Enc_1])

    Enc_1 = Conv2D(filters=first_layer_channels * 2, kernel_size=1, strides=1)(Enc_1)
    Enc_1 = BatchNormalization()(Enc_1)
    Enc_1 = Activation('relu')(Enc_1)

    # Encoder 2
    branch_1_Enc_2 = Conv2D(filters=first_layer_channels * 2, kernel_size=2, strides=2)(Enc_1)
    branch_1_Enc_2 = BatchNormalization()(branch_1_Enc_2)
    branch_1_Enc_2 = Activation('relu')(branch_1_Enc_2)

    branch_1_Enc_2 = Conv2D(filters=first_layer_channels * 2, kernel_size=3, strides=1, padding='same')(branch_1_Enc_2)
    branch_1_Enc_2 = BatchNormalization()(branch_1_Enc_2)
    branch_1_Enc_2 = Activation('relu')(branch_1_Enc_2)

    branch_2_Enc_2 = Conv2D(filters=first_layer_channels * 2, kernel_size=2, strides=2)(Enc_1)
    branch_2_Enc_2 = BatchNormalization()(branch_2_Enc_2)
    branch_2_Enc_2 = Activation('relu')(branch_2_Enc_2)

    branch_2_Enc_2 = Conv2D(filters=first_layer_channels * 2, kernel_size=3, strides=1, padding='same')(branch_2_Enc_2)
    branch_2_Enc_2 = BatchNormalization()(branch_2_Enc_2)
    branch_2_Enc_2 = Activation('relu')(branch_2_Enc_2)

    Enc_2 = concatenate([branch_1_Enc_2, branch_2_Enc_2])

    Enc_2 = Conv2D(filters=first_layer_channels * 4, kernel_size=1, strides=1)(Enc_2)
    Enc_2 = BatchNormalization()(Enc_2)
    Enc_2 = Activation('relu')(Enc_2)

    # Encoder 3
    branch_1_Enc_3 = Conv2D(filters=first_layer_channels * 4, kernel_size=2, strides=2)(Enc_2)
    branch_1_Enc_3 = BatchNormalization()(branch_1_Enc_3)
    branch_1_Enc_3 = Activation('relu')(branch_1_Enc_3)

    branch_1_Enc_3 = Conv2D(filters=first_layer_channels * 4, kernel_size=3, strides=1, padding='same')(branch_1_Enc_3)
    branch_1_Enc_3 = BatchNormalization()(branch_1_Enc_3)
    branch_1_Enc_3 = Activation('relu')(branch_1_Enc_3)

    branch_2_Enc_3 = Conv2D(filters=first_layer_channels * 4, kernel_size=2, strides=2)(Enc_2)
    branch_2_Enc_3 = BatchNormalization()(branch_2_Enc_3)
    branch_2_Enc_3 = Activation('relu')(branch_2_Enc_3)

    branch_2_Enc_3 = Conv2D(filters=first_layer_channels * 4, kernel_size=3, strides=1, padding='same')(branch_2_Enc_3)
    branch_2_Enc_3 = BatchNormalization()(branch_2_Enc_3)
    branch_2_Enc_3 = Activation('relu')(branch_2_Enc_3)

    Enc_3 = concatenate([branch_1_Enc_3, branch_2_Enc_3])

    Enc_3 = Conv2D(filters=first_layer_channels * 8, kernel_size=1, strides=1)(Enc_3)
    Enc_3 = BatchNormalization()(Enc_3)
    Enc_3 = Activation('relu')(Enc_3)

    # Encoder 4
    branch_1_Enc_4 = Conv2D(filters=first_layer_channels * 8, kernel_size=2, strides=2)(Enc_3)
    branch_1_Enc_4 = BatchNormalization()(branch_1_Enc_4)
    branch_1_Enc_4 = Activation('relu')(branch_1_Enc_4)

    branch_1_Enc_4 = Conv2D(filters=first_layer_channels * 8, kernel_size=3, strides=1, padding='same')(branch_1_Enc_4)
    branch_1_Enc_4 = BatchNormalization()(branch_1_Enc_4)
    branch_1_Enc_4 = Activation('relu')(branch_1_Enc_4)

    branch_2_Enc_4 = Conv2D(filters=first_layer_channels * 8, kernel_size=2, strides=2)(Enc_3)
    branch_2_Enc_4 = BatchNormalization()(branch_2_Enc_4)
    branch_2_Enc_4 = Activation('relu')(branch_2_Enc_4)

    branch_2_Enc_4 = Conv2D(filters=first_layer_channels * 8, kernel_size=3, strides=1, padding='same')(branch_2_Enc_4)
    branch_2_Enc_4 = BatchNormalization()(branch_2_Enc_4)
    branch_2_Enc_4 = Activation('relu')(branch_2_Enc_4)

    Enc_4 = concatenate([branch_1_Enc_4, branch_2_Enc_4])

    Enc_4 = Conv2D(filters=first_layer_channels * 16, kernel_size=1, strides=1)(Enc_4)
    Enc_4 = BatchNormalization()(Enc_4)
    Enc_4 = Activation('relu')(Enc_4)

    # Decoder
    # Decoder 1

    branch_1_Dec_1 = Conv2DTranspose(filters=first_layer_channels * 4, kernel_size=2, strides=2)(Enc_4)
    branch_1_Dec_1 = BatchNormalization()(branch_1_Dec_1)
    branch_1_Dec_1 = Activation('relu')(branch_1_Dec_1)

    branch_1_Dec_1 = Conv2DTranspose(filters=first_layer_channels * 4, kernel_size=3, strides=1, padding='same')(
        branch_1_Dec_1)
    branch_1_Dec_1 = BatchNormalization()(branch_1_Dec_1)
    branch_1_Dec_1 = Activation('relu')(branch_1_Dec_1)

    branch_2_Dec_1 = Conv2DTranspose(filters=first_layer_channels * 4, kernel_size=2, strides=2)(Enc_4)
    branch_2_Dec_1 = BatchNormalization()(branch_2_Dec_1)
    branch_2_Dec_1 = Activation('relu')(branch_2_Dec_1)

    branch_2_Dec_1 = Conv2DTranspose(filters=first_layer_channels * 4, kernel_size=3, strides=1, padding='same')(
        branch_2_Dec_1)
    branch_2_Dec_1 = BatchNormalization()(branch_2_Dec_1)
    branch_2_Dec_1 = Activation('relu')(branch_2_Dec_1)

    Dec_1 = concatenate([branch_1_Dec_1, branch_2_Dec_1])

    Dec_1 = concatenate([Dec_1, Enc_3])  ####        quisiamo
    Dec_1 = Conv2D(filters=first_layer_channels * 8, kernel_size=1, strides=1)(Dec_1)
    Dec_1 = BatchNormalization()(Dec_1)
    Dec_1 = Activation('relu')(Dec_1)

    # Decoder 2

    branch_1_Dec_2 = Conv2DTranspose(filters=first_layer_channels * 2, kernel_size=2, strides=2)(Dec_1)
    branch_1_Dec_2 = BatchNormalization()(branch_1_Dec_2)
    branch_1_Dec_2 = Activation('relu')(branch_1_Dec_2)

    branch_1_Dec_2 = Conv2DTranspose(filters=first_layer_channels * 2, kernel_size=3, strides=1, padding='same')(
        branch_1_Dec_2)
    branch_1_Dec_2 = BatchNormalization()(branch_1_Dec_2)
    branch_1_Dec_2 = Activation('relu')(branch_1_Dec_2)

    branch_2_Dec_2 = Conv2DTranspose(filters=first_layer_channels * 2, kernel_size=2, strides=2)(Dec_1)
    branch_2_Dec_2 = BatchNormalization()(branch_2_Dec_2)
    branch_2_Dec_2 = Activation('relu')(branch_2_Dec_2)

    branch_2_Dec_2 = Conv2DTranspose(filters=first_layer_channels * 2, kernel_size=3, strides=1, padding='same')(
        branch_2_Dec_2)
    branch_2_Dec_2 = BatchNormalization()(branch_2_Dec_2)
    branch_2_Dec_2 = Activation('relu')(branch_2_Dec_2)

    Dec_2 = concatenate([branch_1_Dec_2, branch_2_Dec_2])

    Dec_2 = concatenate([Dec_2, Enc_2])
    Dec_2 = Conv2D(filters=first_layer_channels * 4, kernel_size=1, strides=1)(Dec_2)
    Dec_2 = BatchNormalization()(Dec_2)
    Dec_2 = Activation('relu')(Dec_2)

    # Decoder 3

    branch_1_Dec_3 = Conv2DTranspose(filters=first_layer_channels, kernel_size=2, strides=2)(Dec_2)
    branch_1_Dec_3 = BatchNormalization()(branch_1_Dec_3)
    branch_1_Dec_3 = Activation('relu')(branch_1_Dec_3)

    branch_1_Dec_3 = Conv2DTranspose(filters=first_layer_channels, kernel_size=3, strides=1, padding='same')(
        branch_1_Dec_3)
    branch_1_Dec_3 = BatchNormalization()(branch_1_Dec_3)
    branch_1_Dec_3 = Activation('relu')(branch_1_Dec_3)

    branch_2_Dec_3 = Conv2DTranspose(filters=first_layer_channels, kernel_size=2, strides=2)(Dec_2)
    branch_2_Dec_3 = BatchNormalization()(branch_2_Dec_3)
    branch_2_Dec_3 = Activation('relu')(branch_2_Dec_3)

    branch_2_Dec_3 = Conv2DTranspose(filters=first_layer_channels, kernel_size=3, strides=1, padding='same')(
        branch_2_Dec_3)
    branch_2_Dec_3 = BatchNormalization()(branch_2_Dec_3)
    branch_2_Dec_3 = Activation('relu')(branch_2_Dec_3)

    Dec_3 = concatenate([branch_1_Dec_3, branch_2_Dec_3])
    Dec_3 = Conv2D(filters=first_layer_channels * 2, kernel_size=1, strides=1)(Dec_3)
    Dec_3 = BatchNormalization()(Dec_3)
    Dec_3 = Activation('relu')(Dec_3)

    # Decoder 4

    branch_1_Dec_4 = Conv2DTranspose(filters=int(first_layer_channels / 2), kernel_size=2, strides=2)(Dec_3)
    branch_1_Dec_4 = BatchNormalization()(branch_1_Dec_4)
    branch_1_Dec_4 = Activation('relu')(branch_1_Dec_4)

    branch_1_Dec_4 = Conv2DTranspose(filters=int(first_layer_channels / 2), kernel_size=3, strides=1, padding='same')(
        branch_1_Dec_4)
    branch_1_Dec_4 = BatchNormalization()(branch_1_Dec_4)
    branch_1_Dec_4 = Activation('relu')(branch_1_Dec_4)

    branch_2_Dec_4 = Conv2DTranspose(filters=int(first_layer_channels / 2), kernel_size=2, strides=2)(Dec_3)
    branch_2_Dec_4 = BatchNormalization()(branch_2_Dec_4)
    branch_2_Dec_4 = Activation('relu')(branch_2_Dec_4)

    branch_2_Dec_4 = Conv2DTranspose(filters=int(first_layer_channels / 2), kernel_size=3, strides=1, padding='same')(
        branch_2_Dec_4)
    branch_2_Dec_4 = BatchNormalization()(branch_2_Dec_4)
    branch_2_Dec_4 = Activation('relu')(branch_2_Dec_4)

    Dec_4 = concatenate([branch_1_Dec_4, branch_2_Dec_4])
    Dec_4 = Conv2D(filters=first_layer_channels, kernel_size=1, strides=1)(Dec_4)
    Dec_4 = BatchNormalization()(Dec_4)
    Dec_4 = Activation('relu')(Dec_4)

    # Last Step

    Last_Conv_1 = Conv2D(filters=joint_num, kernel_size=1)(Dec_4)
    Last_Conv_1 = BatchNormalization()(Last_Conv_1)

    Last_Conv_2 = Conv2D(filters=compo_num, kernel_size=1)(Dec_4)
    Last_Conv_2 = BatchNormalization()(Last_Conv_2)

    Last_Conv = concatenate([Last_Conv_1, Last_Conv_2])
    Last_Conv_Sigm = Activation('sigmoid')(Last_Conv)

    model = Model(inputs=inputs, outputs=[Last_Conv, Last_Conv_Sigm, Enc_4])

    return model