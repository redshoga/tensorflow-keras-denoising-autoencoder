import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Input, MaxPooling2D, UpSampling2D, Lambda
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from tensorflow.python.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()
# x_train.shape => (60000, 28, 28)
# x_test.shape => (10000, 28, 28)

# CNNで扱いやすいように変換
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_train = x_train/255.
x_test = x_test/255.
# x_train[0].shape => (28, 28, 1)
# x_test[0].shape => (28, 28, 1)

def make_masking_noise_data(data_x, percent=0.1):
    size = data_x.shape
    # maskingはdata_xと同じ形で0or1のみを持つ行列
    masking = np.random.binomial(n=1, p=percent, size=size)
    return data_x*masking

x_train_masked = make_masking_noise_data(x_train)
x_test_masked = make_masking_noise_data(x_test)

def make_gaussian_noise_data(data_x, scale=0.8):
    # 正規分布で乱数を発生して足しこむ
    gaussian_data_x = data_x + np.random.normal(loc=0, scale=scale, size=data_x.shape)
    # clipで0以上1未満にする
    gaussian_data_x = np.clip(gaussian_data_x, 0, 1)
    return gaussian_data_x

x_train_gauss = make_gaussian_noise_data(x_train)
x_test_gauss = make_gaussian_noise_data(x_test)

# サンプルを出力
array_to_img(x_train[0]).save('x_train_sample.png')
array_to_img(x_train_masked[0]).save('x_train_masked_sample.png')
array_to_img(x_train_gauss[0]).save('x_train_gauss_sample.png')

# CNNの構成
autoencoder = Sequential()

# Encoder
autoencoder.add(Conv2D(16, (3, 3), 1, activation='relu', padding='same', input_shape=(28, 28, 1)))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Conv2D(8, (3, 3), 1, activation='relu', padding='same'))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))

# Decoder
autoencoder.add(Conv2D(8, (3, 3), 1, activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(16, (3, 3), 1, activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))

# 出力データのチャンネルを1にするため畳み込み
autoencoder.add(Conv2D(1, (3, 3), 1, activation='sigmoid', padding='same'))
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
initial_weights = autoencoder.get_weights()
autoencoder.summary()

# ガウシアンノイズデータを用いた学習と予測
autoencoder.fit(
    x_train_gauss, # 入力
    x_train, # 正解
    epochs=10,
    batch_size=20,
    shuffle=True
)

gauss_preds = autoencoder.predict(x_test_gauss)
for i in range(10):
    array_to_img(x_test[i]).save('x_test_%d.png' % i)
    array_to_img(x_test_gauss[i]).save('x_test_gauss_%d.png' % i)
    array_to_img(gauss_preds[i]).save('x_test_gauss_pred_%d.png' % i)

# 初期化
autoencoder.set_weights(initial_weights)

# マスキングノイズデータを用いた学習と予測
autoencoder.fit(
    x_train_masked, # 入力
    x_train, # 正解
    epochs=10,
    batch_size=20,
    shuffle=True
)
masked_preds = autoencoder.predict(x_test_masked)
for i in range(10):
    array_to_img(x_test_masked[i]).save('x_test_masked_%d.png' % i)
    array_to_img(masked_preds[i]).save('x_test_masked_pred_%d.png' % i)
