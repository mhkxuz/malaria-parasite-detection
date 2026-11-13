import tensorflow as tf
from tensorflow.keras import layers, models


def build_mobilenet(input_shape=(128, 128, 3)):
    inputs = layers.Input(shape=input_shape)

    # MobileNetV2 expects pixel values in [-1, 1].
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1,
    )(inputs)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(
        inputs,
        outputs,
        name="mobilenetv2_malaria",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model
