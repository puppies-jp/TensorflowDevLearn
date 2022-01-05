# 01\_画像の分類

- [tutorial1](#1)

  - [x] Conv2D とプーリング層で畳み込みニューラル ネットワークを定義する。
  - [x] リアルワールド画像のデータセットを処理するモデルを構築して訓練する。
  - [x] 畳み込みを使用してニューラル ネットワークを改善する方法について理解している。
    - おそらくデータの`拡張`、`dropout`などの`under fitting`, `over fitting` の対策についての問題
      - [00\_モデルの構築と訓練](../00_モデルの構築と訓練)にまとめる。

- [tutorial2](#2)

  - [ ] さまざまな形状やサイズのリアルワールド画像を使用する。

    - [ ] 画素数、channel 数の違うデータの扱いのこと？

  - [ ] 画像の拡張を使用して過剰適合を回避する。
  - [ ] ImageDataGenerator を使用する。
  - [ ] ImageDataGenerator を使用して、ディレクトリ構造に基づいて画像にラベルを付ける方法について理解している。

## <a name="1">tutorial1</a>

### model の構築からコンパイルまで

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# -------------------------------------------- #
# 🌟 Step1 modelの構築 方法

# 🌟🌟 モデルの定義1(後からaddしていく方法)
model = models.Sequential()

# Convレイヤー
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Poolingレイヤー
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# --------------------------- #
# 🌟🌟 モデルの定義2(一括で定義する方法)
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# -------------------------------------------- #
# 🌟 Step2　コンパイル(だいたいこんな感じ)
# 作成したモデルはこのままでもpredictはできるが、fittingしてトレーニングをする必要がある。
model.compile(
    # 最適化関数
    optimizer='adam',
    # 損失関数
    loss='sparse_categorical_crossentropy',
    # watchするパラメータ
    metrics=['accuracy'])

# -------------------------------------------- #
# 🌟 Step3 fitting
# ポイントとしては、バッチサイズ、epoch数、
# training dataとvalidation data　を使うかといった観点を考える必要がある。

# これ
model.fit(train_images, train_labels, epochs=5)

# または こんな感じ
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

# -------------------------------------------- #
# Step4 🌟モデルを動かして結果を得る

# モデルの評価をしてくれる関数 戻り値からmetricを取得できる。
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# まぁ、こんな感じ　 input_shape = (x,y,z)に対して入力するデータ数をnとすると、
# (n, x, y, z)が入力となるように整える。
# 分類の場合、各クラスの確率となるのでindexの最大値を取る
num = 10
res = model.predict(test_images[:num])

for i in range(len(res)):
    print(res[i].argmax() , " == " , test_labels[i]," : ",res[i].argmax() == test_labels[i])
```

## <a name="2">tutorial2</a>

```python

```
