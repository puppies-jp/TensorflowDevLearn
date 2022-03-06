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
