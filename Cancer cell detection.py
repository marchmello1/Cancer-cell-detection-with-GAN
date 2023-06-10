import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from PIL import Image
from google.colab import drive
from tensorflow.keras import models, layers
import os

drive.mount('/drive')

metadata_file_path = '/drive/MyDrive/HAM10000_metadata.csv'
dataset_folder = '/drive/MyDrive/HAM10000_images_part_1'

data = pd.read_csv(metadata_file_path)

data = data.drop(columns=['lesion_id', 'image_id', 'dx_type', 'age', 'sex', 'localization'])
label_encoder = LabelEncoder()
data['dx'] = label_encoder.fit_transform(data['dx'])

subset_size = 2100
image_files = os.listdir(dataset_folder)[:subset_size]
real_images = []
for file in image_files:
    image_path = dataset_folder + '/' + file
    image = load_img(image_path, target_size=(32, 32))
    image_array = img_to_array(image)
    real_images.append(image_array)

real_images = np.array(real_images)
real_images = real_images / 255.0
real_images = real_images.reshape(-1, 32, 32, 3)

generated_images = np.random.normal(0, 1, size=(subset_size, 100))

generator = models.Sequential()
generator.add(layers.Dense(128 * 8 * 8, input_dim=100))
generator.add(layers.BatchNormalization())
generator.add(layers.LeakyReLU(0.2))
generator.add(layers.Reshape((8, 8, 128)))
generator.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
generator.add(layers.BatchNormalization())
generator.add(layers.LeakyReLU(0.2))
generator.add(layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='sigmoid'))

discriminator = models.Sequential()
discriminator.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=(32, 32, 3)))
discriminator.add(layers.LeakyReLU(0.2))
discriminator.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
discriminator.add(layers.LeakyReLU(0.2))
discriminator.add(layers.Flatten())
discriminator.add(layers.Dense(1, activation='sigmoid'))

discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
discriminator.trainable = False

gan_input = layers.Input(shape=(100,))
gan_output = discriminator(generator(gan_input))
gan = models.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

batch_size = 128
epochs = 20
steps_per_epoch = real_images.shape[0] // batch_size

for epoch in range(epochs):
    for step in range(steps_per_epoch):
        noise = np.random.normal(0, 1, size=(batch_size, 100))
        generated_images = generator.predict(noise)
        real_images_batch = real_images[np.random.randint(0, real_images.shape[0], size=batch_size)]
        X = np.concatenate([real_images_batch, generated_images])
        y = np.zeros(2 * batch_size)
        y[:batch_size] = 1
        discriminator.trainable = True
        discriminator.train_on_batch(X, y)
        noise = np.random.normal(0, 1, size=(batch_size, 100))
        y_gen = np.ones(batch_size)
        discriminator.trainable = False
        gan.train_on_batch(noise, y_gen)
    print(f"Epoch {epoch + 1}/{epochs} completed")

test_noise = np.random.normal(0, 1, size=(real_images.shape[0], 100))
generated_images = generator.predict(test_noise)
test_labels = np.zeros(real_images.shape[0])
test_loss, test_acc = discriminator.evaluate(generated_images, test_labels)
print('Test accuracy:', test_acc)
