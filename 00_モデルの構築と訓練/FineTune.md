# 転移学習,FineTune,パラメータチューニング

- [tutorial2](#tutorial2)
  - [x] 事前訓練されたモデルを使用する(転移学習)。
  - [x] 事前訓練されたモデルから機能を抽出する。
  - [x] モデルへの入力が適切な形状で行われるようにする。
  - [x] テストデータをニューラルネットワークの入力の形状に合わせたものにする。
  - [x] ニューラル ネットワークの出力データを、テストデータで指定された入力の形状に合わせたものにする。

- [転移学習](#Transfer_FineTune)
- [ハイパーパラメータチューニング](#Parameter)

## <a name=Transfer_FineTune> 転移学習 FineTune</a>

- [転移学習](#Transfer)
- [FineTune](#FineTune)

### <a name=Transfer>転移学習</a>

```python
# Create the base model from the pre-trained model MobileNet V2

"""🌟 これで(160,160) ⇨ (160,160,3) になる
"""
IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    # なぜtopか?
    # 機械学習モデルの図では、下から上に向かうため、「上」に分類レイヤーがある。
    # 転移／FineTuneでは自身が使用する分には不要な分類が
    # 含まれるためinclude_top=False にすることで不要な層を省くようにしている。
    include_top=False, # 🌟上位の分類レイヤーを含まない
    # 🌟imageNet(多彩なカテゴリを持つ研究用トレーニングデータセット)
    weights='imagenet' 
    )
```

```python
""" 🌟 このようにすることで、取得したモデルをトレーニングによって変更できなくする。
"""
base_model.trainable = False
```

```python

"""🌟 こんな感じでデータ拡張層をモデルに組み込むことができる。
     trainingでのみ有効になるらしい
"""
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

"""🌟 MobileNetはピクセル値[-1,1]を想定したモデルのため、
    　このようにしてmodelのレイヤーに入れることで、リスケールできる。
"""
# 🌟 これでも代用可
# rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs) # データ拡張層
x = preprocess_input(x) # 入力層
# 🌟　MobileNet(出力は(5, 5, 1280)で出力される
x = base_model(x, training=False) 
x = global_average_layer(x) # pooling層
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x) # 出力層
model = tf.keras.Model(inputs, outputs)
```

### <a name=FineTune>FineTune</a>

- 転移学習でパフォーマンスをさらに向上させる方法の 1 つに、追加した分類器のトレーニングと並行して、事前トレーニング済みモデルの最上位レイヤーの重みをトレーニング（または「ファインチューニング」）するというものがあります。

  - 🌟 **事前トレーニング済みモデルをトレーニング不可に設定し、最上位の分類器をトレーニングした後に行うようにしてください。** 事前トレーニング済みモデルの上にランダムに初期化された分類器を追加してすべてのレイヤーを結合トレーニングしようとすると、
  `（分類器からのランダムな重みにより）`
  **`勾配の更新規模が大きすぎて`**、
  `事前トレーニング済みモデルが学習したことを忘れてしまいます`

```python

"""🌟 ここをtrueにすることで、
    　base_modelをトレーニングできる状態にする。 
"""
base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print(
  "Number of layers in the base model: ", 
  # 🌟 こうすることで、モデルのレイヤー数が確認できる。
  len(base_model.layers))

"""🌟 下のレイヤー(入力層)側から100のレイヤー
      までの重みを変更できなくする。
"""
# Fine-tune from this layer onwards
fine_tune_at = 100 

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

"""🌟 設定後はビルドすること
"""
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])

```

- 学習できるレイヤーは以下のようにして調べられる

```python
len(model.trainable_variables)
```

## <a name=Parameter>Hyperparameters tuning</a>
