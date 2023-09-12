import tensorflow as tf

def create_lenet5(input_shape, num_classes):
    # Model layers
    lenet5 = tf.keras.Sequential([
            # Reshape layer
            tf.keras.layers.Reshape(input_shape, input_shape=(28, 28)),  # Example input shape, change as needed
            # Layer 1 Conv2D
            tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='same'),
            # Layer 2 Pooling Layer
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            # Layer 3 Conv2D
            tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'),
            # Layer 4 Pooling Layer
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            # Flatten
            tf.keras.layers.Flatten(),
            # Layer 5 Dense
            tf.keras.layers.Dense(units=120, activation='tanh'),
            # Layer 6 Dense
            tf.keras.layers.Dense(units=84, activation='tanh'),
            # Layer 7 Dense
            tf.keras.layers.Dense(units=num_classes, activation='softmax')  # Example num_classes=10, change as needed
    ])

    lenet5.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')],
    optimizer = tf.keras.optimizers.Adam()
    )

    return lenet5, lenet5.loss, lenet5.optimizer