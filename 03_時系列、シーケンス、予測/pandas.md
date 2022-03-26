# pandas 便利な使い方まとめ

## 日付の文字列を datetime に変換する

```python
df = pd.read_csv(csv_path)
# Slice [start:stop:step], starting from index 5 take every 6th record.
df = df[5::6]

# 🌟 こんな感じですることで、
# 文字列をdatetimeで読み込むことができる。
date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

# 🌟　こうすることで、datetimeのクラスから、floatのtimestamp(seconds)に変換できる
timestamp_s = date_time.map(pd.Timestamp.timestamp)
```

## グラフ作成系

- index 系列でグラフを書く

```python
plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
plot_features = df[plot_cols]
plot_features.index = date_time

## こう書くことで、subplotでグラフをかける
_ = plot_features.plot(subplots=True)
```

![subpolot](pandas_subplot.png)

- heatmap を作成する。

```python
# hist2d でheatmapっぽいのが作れるのか。。結構便利かも
plt.hist2d(df['wd (deg)'], df['wv (m/s)'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('Wind Direction [deg]')
plt.ylabel('Wind Velocity [m/s]')
```

![heatmap](wind_heatmap.png)

## 相関係数のヒートマップを作成

```python
def show_heatmap(data):
    # 🌟マップにする
    plt.matshow(
        ## 🌟相関係数を導出
        data.corr()
        )
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()


show_heatmap(df)
```

![相関係数heatmap](corrHeatmap.png)
