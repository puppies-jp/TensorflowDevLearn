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

リンク

- [データセット作成](#dataset)
- [フォルダ構成ごとにまとまったデータセット](#DirDataset)
- [モデル作成(基本的)](#basic_model)
- [RNN適用モデル](#rnn_model)
- [BERTモデル](#BERT_model)

---

## <a name=dataset>文字列データセットの作成方法</a>

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

## <a name=DirDataset>データセット作成方法2</a>

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
    # 🌟datasetの最大文字列長(指定しないと最大文字列長に合わせられる)
    output_sequence_length=sequence_length　
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

## <a name=basic_model>モデルの作成</a>

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

## <a name=rnn_model>RNNを適用したモデル</a>

- RNN層1つだけのモデル

```python
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
```

- RNNを2層スタックしたモデル
  - 🌟`return_sequences`の使い方に注目
    - true: 毎回入力ごとにreturnする
    - false: 最後に出力する。

```python
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
    # 🌟return_sequences=trueとすることで毎回returnするようになっている。
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])
```

## <a name=BERT_model>BERTを使用したモデル</a>

- そもそもBERTとは、、、

  > BERT and other Transformer encoder architectures have been wildly successful on a variety of tasks in NLP (natural language processing). 
  > They compute vector-space representations of natural language that are suitable for use in deep learning models. 
  > The BERT family of models uses the Transformer encoder architecture to process each token of input text in the full context of all tokens before and after, hence the name: Bidirectional Encoder Representations from Transformers.

  - つまりエンコーダーみたいなもの？

  - BERTモデルの読み込み
    - tfhubから読み込みます。

        <details><summary>(▶︎click it)BERTモデル群(これらから重みを選べる)</summary><div>

        <pre><code>
        bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8' 

        map_name_to_handle = {
            'bert_en_uncased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
            'bert_en_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
            'bert_multi_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
            'small_bert/bert_en_uncased_L-2_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-2_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-2_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-2_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-4_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-4_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-4_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-4_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-6_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-6_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-6_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-6_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-8_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-8_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-8_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-8_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-10_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-10_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-10_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-10_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-12_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-12_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-12_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
            'albert_en_base':
                'https://tfhub.dev/tensorflow/albert_en_base/2',
            'electra_small':
                'https://tfhub.dev/google/electra_small/2',
            'electra_base':
                'https://tfhub.dev/google/electra_base/2',
            'experts_pubmed':
                'https://tfhub.dev/google/experts/bert/pubmed/2',
            'experts_wiki_books':
                'https://tfhub.dev/google/experts/bert/wiki_books/2',
            'talking-heads_base':
                'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
        }

        map_model_to_preprocess = {
            'bert_en_uncased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'bert_en_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
            'small_bert/bert_en_uncased_L-2_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-2_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-2_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-2_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-4_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-4_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-4_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-4_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-6_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-6_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-6_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-6_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-8_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-8_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-8_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-8_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-10_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-10_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-10_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-10_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-12_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-12_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-12_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'bert_multi_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
            'albert_en_base':
                'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
            'electra_small':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'electra_base':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'experts_pubmed':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'experts_wiki_books':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'talking-heads_base':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        }

        tfhub_handle_encoder = map_name_to_handle[bert_model_name]
        tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

        print(f'BERT model selected           : {tfhub_handle_encoder}')
        print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')
        </code></pre>

        </div></details>

    - 🌟基本これを選んどけばOK

        ```python
        tfhub_handle_encoder = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1"
        tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

        # 🌟🌟🌟　モデルの読み込み　🌟🌟🌟 
        ## 🌟プリプロセスモデル(あくまでpreprocessモデル)
        bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

        # 🌟ざっくりした使い方 
        text_test = ['this is such an amazing movie!']
        text_preprocessed = bert_preprocess_model(text_test)

        print(f'Keys       : {list(text_preprocessed.keys())}')
        print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
        print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
        print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
        print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

        # 🌟🌟🌟BERTモデルの読み込み🌟🌟🌟
        ## 🌟 あくまでここで読み込んでいることに注意
        bert_model = hub.KerasLayer(tfhub_handle_encoder)
        
        # 🌟🌟使ってみる
        # プリプロセスモデルの出力結果を渡していることに注意
        bert_results = bert_model(text_preprocessed)

        print(f'Loaded BERT: {tfhub_handle_encoder}')
        print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
        print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
        print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
        print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')
        ```

