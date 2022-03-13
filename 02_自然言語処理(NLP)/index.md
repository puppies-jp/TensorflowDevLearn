# [02\_自然言語処理（NLP）](<02_自然言語処理(NLP)>)

- [ ] TensorFlow を使った自然言語処理システムを構築する。
- [x] TensorFlow で使用するテキストを用意する。
- [x] 二項分類を使用してテキストのカテゴリを特定するモデルを構築する
- [ ] 多項分類を使用してテキストのカテゴリを特定するモデルを構築する
- [x] TensorFlow モデルで単語埋め込みを使用する。
- [ ] 二項分類または多項分類のいずれかのモデルで、LSTM を使用してテキストを分類する。
- [ ] モデルに RNN 層と GRU 層を追加する。
- [ ] テキストを処理するモデルで、RNN、GRU、CNN を使用する。
- [ ] LSTM を既存のテキストで訓練して、テキストを生成する(歌や詩など)

---

## 文字列データセットの作成方法

- 以下に手順をまとめる

    1. 文字列を整数にエンコードした配列にし、単語のマップを作成する

    1. 配列をワンホット（one-hot）エンコーディングする。

    1. 配列をパディングによって同じ長さに揃え、`(サンプル数 * 許容できる長さの最大値)` の形の整数テンソルに変更する。

        1. 数値の配列をシンプルにpaddingする方法

            ```python
            train_data = 
                keras.preprocessing.sequence.pad_sequences(
                    train_data,
                    value=word_index["<PAD>"],
                    padding='post',
                    maxlen=256
                    )
                    
            test_data = 
                keras.preprocessing.sequence.pad_sequences(
                    test_data,
                    value=word_index["<PAD>"],
                    padding='post',
                    maxlen=256
                    )
            ```

        2. 🌟(ここ覚えとくこと！！)textから変換する方法 

            ```python
            def custom_standardization(input_data):
                lowercase = tf.strings.lower(input_data)
                
                # 🌟 <br>タグをけす。
                stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')

                # 🌟 punctuation == 文字列の句読点
                # 🌟 re.escape(pattern)
                #   pattern 中の特殊文字をエスケープします。
                #   これは正規表現メタ文字を含みうる任意のリテラル文字列にマッチしたい時に便利です。     
                return tf.strings.regex_replace(
                    stripped_html,
                    '[%s]' % re.escape(string.punctuation),
                    ''
                    )


            max_features = 10000
            sequence_length = 250
            vectorize_layer = layers.TextVectorization(
                standardize=custom_standardization,
                max_tokens=max_features,
                output_mode='int',
                output_sequence_length=sequence_length
                )
            
            # Make a text-only dataset (without labels), then call adapt
            train_text = raw_train_ds.map(lambda x, y: x)
            vectorize_layer.adapt(train_text)

            # 🌟 textをvector化してlabelと一緒に出力する。
            def vectorize_text(text, label):
                text = tf.expand_dims(text, -1)
                return vectorize_layer(text), label

            # 🌟 使用例1 :textとlabelからVectorとlabelを生成する。
            # retrieve a batch (of 32 reviews and labels) from the dataset
            text_batch, label_batch = next(iter(raw_train_ds))
            first_review, first_label = text_batch[0], label_batch[0]
            print("Review", first_review)
            print("Label", raw_train_ds.class_names[first_label])
            print("Vectorized review", vectorize_text(first_review, first_label))

            # 🌟 datasetを作成する。
            # 🌟 tf.keras.utils.text_dataset_from_directoryにて作成したdataset
            train_ds = raw_train_ds.map(vectorize_text)
            val_ds = raw_val_ds.map(vectorize_text)
            test_ds = raw_test_ds.map(vectorize_text)

            ```

## データセット作成方法2

- データセットの構成を読み込む
  - こんな感じのファイル構成で各ディレクトリにテキストが入っている構成を前提に考える

```sh
user@MacBook text % tree -d aclImdb
aclImdb
├── test # 検証用データセット
│   ├── neg # label1
│   └── pos # label2
└── train  # 学習用データセット
    ├── neg
    └── pos

7 directories
```

1. `text_dataset_from_directory`を使う
こうすることでtrainからデータセットを読み込んでくれる。

```python
batch_size = 32
seed = 42

# 🌟trainデータセット
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', 
    seed=seed)

# 🌟validation データセット
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', 
    seed=seed)

for index in range(len(raw_train_ds.class_names)):
    # 🌟 indexとラベルが対応づけられてる。
    print(f"Label {index} corresponds to {raw_train_ds.class_names[index]}")
    
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

# 🌟ここのパラメータを共通化しないとわけがわからなくなる
max_features = 10000　# 🌟単語数
sequence_length = 250 # 🌟datasetの最大文字列長

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,　# 🌟単語数
    output_mode='int',
    output_sequence_length=sequence_length　# 🌟datasetの最大文字列長
    )

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label 

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)

# 🌟cashe,prefchを設定する。
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

## モデルの作成

```python
embedding_dim = 16

model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

model.summary()
```

```python
# 🌟🌟modelの作り方2🌟🌟
export_model = tf.keras.Sequential([
  ## 🌟ここにvectorize層を追加することで直接的に文字列を入力できるようになる。
  vectorize_layer, 
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), 
    optimizer="adam", 
    metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)

examples = [
  "The movie was great!",
  "The movie was okay.",
  "The movie was terrible..."
]

export_model.predict(examples)

```