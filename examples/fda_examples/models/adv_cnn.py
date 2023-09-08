import tensorflow as tf

def create_adv_cnn(input_shape, num_classes):
    # Model layers
    adv_cnn = tf.keras.Sequential([
        # Reshape layer
        tf.keras.layers.Reshape(input_shape, input_shape=(28, 28)),  # Example input shape, change as needed
        # First Convolutional Block
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Second Convolutional Block
        tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Third Convolutional Block
        tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Fully Connected Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    adv_cnn.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')]
    )

    return adv_cnn, adv_cnn.loss