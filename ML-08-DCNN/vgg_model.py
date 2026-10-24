import tensorflow as tf
from tensorflow.keras import layers, models

def create_vgg_config_1(input_shape, num_classes):
    """Configuration 1: Transfer Learning ด้วย VGG16 แบบ Frozen Base"""
    base_model = tf.keras.applications.VGG16(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="VGG16_Config1")
    return model

def create_vgg_config_2(input_shape, num_classes):
    """Configuration 2: Deep VGG16 with Dropout & Larger Dense Layer"""
    base_model = tf.keras.applications.VGG16(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="VGG16_Config2")
    return model