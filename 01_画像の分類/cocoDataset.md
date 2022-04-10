# 物体検出データセット

## COCO_2017

### 取得方法

```python
import zipfile

import tensorflow as tf
from tensorflow import keras

url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
filename = os.path.join(os.getcwd(), "data.zip")
filepath = keras.utils.get_file(filename, url)


with zipfile.ZipFile(filepath, "r") as z_fp:
    z_fp.extractall("./")

```

### 読み込み方

```python
import tensorflow_datasets as tfds

(train_dataset, val_dataset), dataset_info = tfds.load(
    "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
)
```
