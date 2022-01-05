# 00\_モデルの構築と訓練

## 学ぶべき項目は以下

- TensorFlow2.0 以降を使用する。

- [tutorial1](#tutorial1)

  - [ ] TensorFlow を使用して機械学習(ML)モデルの構築、コンパイル、訓練を行う。
  - [ ] データを前処理してモデルで使用できるようにする。
  - [ ] モデルを使用して結果を予測する。
  - [ ] 複数の層で構成されるシーケンスモデルを構築する。
  - [ ] 二項分類のモデルを構築して訓練する。
  - [ ] 多項分類のモデルを構築して訓練する。

- [tutorial2](#tutorial2)

  - [ ] 事前訓練されたモデルを使用する(転移学習)。
  - [ ] 事前訓練されたモデルから機能を抽出する。
  - [ ] モデルへの入力が適切な形状で行われるようにする。
  - [ ] テストデータをニューラルネットワークの入力の形状に合わせたものにする。
  - [ ] ニューラル ネットワークの出力データを、テストデータで指定された入力の形状に合わせたものにする。

- [tutorial3](#tutorial3)

  - [ ] データの一括読み込みについて理解している
  - [ ] コールバックを使用して、訓練サイクルの終了を呼び出す。
  - [ ] 複数のソースのデータセットを使用する。
  - [ ] 複数のフォーマット(json や csv など)のデータセットを使用する。
  - [ ] tf.data.datasets のデータセットを使用する。

- [過学習](#OverFitting)

  - [ ] 訓練済みモデルのプロットの損失と精度を確認する。
  - [ ] 拡張やドロップアウトなどの過剰適合を避けるための戦略を割り出す。

---

## <a name="tutorial1">tutorial1</a>

## <a name="tutorial2">tutorial2</a>

## <a name="tutorial3">tutorial3</a>

---

## <a name="OverFitting">過学習</a>

### 過学習の対策に以下が有効と言われている。

- [ ] 訓練データを増やす
- [ ] data 拡張を実行
- [x] [dropout 層追加(**モデル側の改善策**)](#dropOut)
- [x] [network の容量を減らす(**モデル側の改善策**)](#SmallModel)
- [ ] 重みの正則化(L1 正則化、L2 正則化などをして過学習を防ぐ)
- [ ] バッチ正規化

---

- <a name="dropOut">dropout 層追加(**モデル側の改善策**)</a>

  - **訓練時**に層から出力された特徴量に対してランダムに「ドロップアウト（つまりゼロ化）」を行うもの`ドロップアウト率`は概ね 0.2 ~ 0.5 が目安と言われている。

  ```python
  # 🌟 使い方 これをモデルの層に追加していくだけ。
  keras.layers.Dropout(0.5),
  ```

- <a name="SmallModel">network の容量を減らす(**モデル側の改善策**)</a>

  - パラメータ数が多いと余計な条件に引っかかるようになり、増えすぎることで汎化性能が下がっていく(test データでは高い正解率にも関わらず、validation check にて低いスコアが出てしまう。(**過学習**))
  - セオリーとして小さいモデル、大きいモデルの丁度いいところを探す必要がある。

- 以下に例を示す。

```python
# -------------------------------------------- #
# 🌟　普通のモデル
baseline_model = keras.Sequential([
    # `.summary` を見るために`input_shape`が必要
    keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

baseline_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])

baseline_model.summary()
baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)

# -------------------------------------------- #
# 🌟　小さいモデル
smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

smaller_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_crossentropy'])

smaller_model.summary()

smaller_history = smaller_model.fit(train_data,
                                    train_labels,
                                    epochs=20,
                                    batch_size=512,
                                    validation_data=(test_data, test_labels),
                                    verbose=2)
# -------------------------------------------- #
# 🌟　大きいモデル
bigger_model = keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

bigger_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy','binary_crossentropy'])

bigger_model.summary()
bigger_history = bigger_model.fit(train_data, train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)
# -------------------------------------------- #


# 🌟 大中小のモデルのbinary_crossentropy,epochを表示する関数
def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16,10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                        '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])

# 🌟モデルのhistryを渡す
plot_history([('baseline', baseline_history),
              ('smaller', smaller_history),
              ('bigger', bigger_history)])

```

- 大、中、小モデルの実行結果

  - Baseline と Bigger の train,val が早々に乖離して過学習を起こしていることが分かる。
    ![OverFit.png](OverFit.png)

- [ ] 重みの正則化(L1 正則化、L2 正則化などをして過学習を防ぐ)
- [ ] バッチ正規化
- [ ] 訓練データを増やす
