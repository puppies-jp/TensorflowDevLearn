# 時系列系データセット形成メモ

- indexのenumマップを作るやつ
  - 結構便利そうなのでメモっとく

```python
# columns名をindexに落とし込んでるだけ。
column_indices = {name: i for i, name in enumerate(df.columns)}

print(df.columns)
pprint(column_indices)
```

- 出力結果(columnsがindex番号付きでマッピングされる)
  - 🌟numpyに変換したときに何のデータかわかるようになる。

```python
Index(['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
       'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
       'H2OC (mmol/mol)', 'rho (g/m**3)', 'Wx', 'Wy', 'max Wx', 'max Wy',
       'Day sin', 'Day cos', 'Year sin', 'Year cos'],
      dtype='object')
{'Day cos': 16,
 'Day sin': 15,
 'H2OC (mmol/mol)': 9,
 'T (degC)': 1,
 'Tdew (degC)': 3,
 'Tpot (K)': 2,
 'VPact (mbar)': 6,
 'VPdef (mbar)': 7,
 'VPmax (mbar)': 5,
 'Wx': 11,
 'Wy': 12,
 'Year cos': 18,
 'Year sin': 17,
 'max Wx': 13,
 'max Wy': 14,
 'p (mbar)': 0,
 'rh (%)': 4,
 'rho (g/m**3)': 10,
 'sh (g/kg)': 8}
```

```python
"""🌟 入出力を整える関数 """
def split_window(self, features):
    print(f"{split_window.__name__}---------------")
    print(f"features(shape):{features.shape}")
    print(f'input_slice: {self.input_slice}')
    print(f'label_slice: {self.labels_slice}')
    print(f'input_width: {self.input_width}')
    print(f'label_width: {self.label_width}')

    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        print(f'column_indices: {self.column_indices}')
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] 
             for name in self.label_columns],axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])
    print("----------------------------")
    return inputs, labels

"""🌟 tensorflow datasetに変換する関数 """
def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
  
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,)

    ds = ds.map(self.split_window)
    return ds

```