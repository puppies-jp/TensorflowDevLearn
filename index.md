# TensorFlow デベロッパー認定資格

```sh
# なんかcondaだと2.8.0でインストールされなかった。。
pip install tensorflow

pip install tensorflow-hub
pip install tensorflow-datasets

pip install -q seaborn
pip install -q -U keras-tuner

# Required to save models in HDF5 format
pip install pyyaml h5py  

pip install -q -U "tensorflow-text==2.8.*"

# AdamW optimizer
pip install -q tf-models-official==2.7.0

# モデル画像生成用(🚨環境によっては動かない)
pip install pydot
```

```python
# dataset周りで頻出するため、ここにメモっておく
AUTOTUNE = tf.data.AUTOTUNE
```

- TensorFlow デベロッパー認定資格の勉強の備忘録として作成した

- 多分使うリンク
  - [転移学習,FineTune,Hyperparameter Tune](00_モデルの構築と訓練/FineTune)
  - [不均衡なデータの検証](imbalanced)
  - [モデルの保存/読み込み](#save_load)
  - [GPU使い方まとめ](GPU)

## [00_TensorFlow デベロッパースキル](00_デベロッパースキル)

- 正直何をメモっていいかわからなからここにかく

  - [x] Python でプログラミングし、Python に関する問題を解決し、PyCharm で Python プログラムをコンパイルして実行する方法について理解している。
  - [ ] TensorFlow API に関する情報を入手する方法（tensorflow.org でガイドや API リファレンスを探す方法な
        ど）ついて理解している。
  - [ ] TensorFlow API のエラー メッセージをデバッグし、調査し、解決する方法について理解している。
        必要に応じて tensorflow.org 以外でも情報を検索して TensorFlow に関する疑問を解決する方法について理解している。
  - [x] 解決しようとしている問題に見合ったモデルサイズの ML モデルを、TensorFlow を使用して作成する方法について理解している。
  - [x] [ML モデルを保存してモデルのファイルサイズを確認する方法について理解している。](#save_load)
    - 必須スキルのため、以下にまとめる
  - [ ] TensorFlow の異なったバージョンによる互換性の違いについて理解している。

## [00\_モデルの構築と訓練](00_モデルの構築と訓練)

- TensorFlow2.0 以降を使用する。
  - [x] TensorFlow を使用して機械学習(ML)モデルの構築、コンパイル、訓練を行う。
  - [ ] データを前処理してモデルで使用できるようにする。
  - [x] モデルを使用して結果を予測する。
  - [x] 複数の層で構成されるシーケンスモデルを構築する。
  - [x] 二項分類のモデルを構築して訓練する。
  - [x] 多項分類のモデルを構築して訓練する。
  - [x] 訓練済みモデルのプロットの損失と精度を確認する。
  - [x] 拡張やドロップアウトなどの過剰適合を避けるための戦略を割り出す。
  - [x] 事前訓練されたモデルを使用する(転移学習)。
  - [ ] 事前訓練されたモデルから機能を抽出する。
  - [x] モデルへの入力が適切な形状で行われるようにする。
  - [ ] テストデータをニューラルネットワークの入力の形状に合わせたものにする。
  - [ ] ニューラル ネットワークの出力データを、テストデータで指定された入力の形状に合わせたものにする。
  - [x] データの一括読み込みについて理解している
  - [x] コールバックを使用して、訓練サイクルの終了を呼び出す。
  - [ ] 複数のソースのデータセットを使用する。
  - [ ] 複数のフォーマット(json や csv など)のデータセットを使用する。
  - [x] tf.data.datasets のデータセットを使用する。

## [01\_画像の分類](01_画像の分類)

- [x] Conv2D とプーリング層で畳み込みニューラル ネットワークを定義する。
- [x] リアルワールド画像のデータセットを処理するモデルを構築して訓練する。
- [x] 畳み込みを使用してニューラル ネットワークを改善する方法について理解している。
- [x] さまざまな形状やサイズのリアルワールド画像を使用する。
- [x] 画像の拡張を使用して過剰適合を回避する。
- [x] ImageDataGenerator を使用する。
- [x] ImageDataGenerator を使用して、ディレクトリ構造に基づいて画像にラベルを付ける方法について理解している。

## [02\_自然言語処理（NLP）](<02_自然言語処理(NLP)>)

- [x] TensorFlow を使った自然言語処理システムを構築する。
- [x] TensorFlow で使用するテキストを用意する。
- [x] 二項分類を使用してテキストのカテゴリを特定するモデルを構築する
- [x] 多項分類を使用してテキストのカテゴリを特定するモデルを構築する
- [x] TensorFlow モデルで単語埋め込みを使用する。
- [x] 二項分類または多項分類のいずれかのモデルで、LSTM を使用してテキストを分類する。
- [x] モデルに RNN 層と GRU 層を追加する。
- [x] テキストを処理するモデルで、RNN、GRU、CNN を使用する。
- [ ] LSTM を既存のテキストで訓練して、テキストを生成する(歌や詩など)

## [03\_時系列、シーケンス、予測](03_時系列、シーケンス、予測)

- [x] 時系列モデル、シーケンス モデル、予測モデルを訓練、調整、使用する。
- [x] 単変量時系列と多変量時系列の値をどちらも予測できるよう訓練する。
- [ ] 時系列学習で使用するデータを準備する。
- [ ] 平均絶対誤差(MAE)と、それを使用してシーケンス モデルの精度を評価する方法について理解する。
- [ ] 時系列モデル、シーケンス モデル、予測モデルで、RNN および CNN を使用する。
- [ ] 追跡ウィンドウまたは中央ウィンドウのどちらを使用すべきかを特定する。

  - 誤訳？

- [ ] TensorFlow を使用して予測を行う。
- [ ] 機能とラベルを準備する。
- [ ] シーケンスのバイアスを特定して補完する。
- [ ] 時系列モデル、シーケンス モデル、予測モデルでの学習率を動的に調整する。

---

## <a name=save_load>モデルの保存/読み込み</a>

### 重みを保存する

1. 訓練の途中あるいは終了後にチェックポイントを自動的に保存する

   ```python
   # pathをセットする。
   checkpoint_path = "training_1/cp.ckpt"
   checkpoint_dir = os.path.dirname(checkpoint_path)

   # チェックポイントコールバックを作る
   cp_callback = tf.keras.callbacks.ModelCheckpoint(
       checkpoint_path,
       save_weights_only=True,
       verbose=1)

   # 新しいコールバックを用いるようモデルを訓練
   model.fit(train_images,
           train_labels,
           epochs=10,
           validation_data=(test_images,test_labels),
           callbacks=[cp_callback])  # 訓練にコールバックを渡す

   # save と loadの前後でacc, lossが同程度であることが確認できる
   loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
   ```

   - これで epoch 数ごとに保存することができる

   ```python
    # ファイル名に(`str.format`を使って)エポック数を埋め込む
    checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # 5エポックごとにモデルの重みを保存するコールバックを作成
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=5)

    # `checkpoint_path` フォーマットで重みを保存
    model.save_weights(checkpoint_path.format(epoch=0))

   ```

1. チェックポイントからロードする。

   ```python
   # modelのインスタンスは save前と同じである必要があるっぽい
   model.load_weights(checkpoint_path)
   # save と loadの前後でacc, lossが同程度であることが確認できる
   loss,acc = model.evaluate(test_images,  test_labels, verbose=2)

   # checkpointディレクトリから最新のチェックポイントのパスを取り出す
   latest = tf.train.latest_checkpoint(checkpoint_dir)
   latest
   ```

1. 手動で保存,ロードする

   ```python
   # 重みの保存
   model.save_weights('./checkpoints/my_checkpoint')

   # 新しいモデルのインスタンスを作成
   model = create_model()

   # 重みの復元
   model.load_weights('./checkpoints/my_checkpoint')
   ```

### モデル全体を保存する

- モデル全体を 2 つの異なるファイルフォーマット (`SavedModel` と `HDF5`) に保存できます。

- モデルの保存(SaveModel)

  ```python
  # モデル全体を SavedModel として保存
  !mkdir -p saved_model
  model.save('saved_model/my_model')
  ```

  - モデルのロード

  ```python
  new_model = tf.keras.models.load_model('saved_model/my_model')

  # アーキテクチャを確認
  new_model.summary()
  ```

- モデルの保存(HDF5)

  ```python
  # HDF5 ファイルにモデル全体を保存
  # 拡張子 '.h5' はモデルが HDF5 で保存されているということを暗示する
  model.save('my_model.h5')
  ```

- モデルのロード

  ```python
  # 同じモデルを読み込んで、重みやオプティマイザーを含むモデル全体を再作成
  new_model = tf.keras.models.load_model('my_model.h5')

  # モデルのアーキテクチャを表示
  new_model.summary()
  ```
