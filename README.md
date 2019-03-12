# CRNN训练及应用

## 1. 语料处理

由甲方提供的

1. 商品列表
2. 商品编号
3. 其他品牌

等数据表被作为学习预料, 凡出现空格的预料都需要换行处理

(处理好的语料存放在`text_renderer/data/corpus/learn_data.txt`)



## 2. 学习样本制作

现实中可以采集到的学习样本较少,故采用合成的方式来制作学习样本.

自动生成器: https://github.com/Sanster/text_renderer.git

训练集生成命令:

``` bash
python text_renderer/main.py # --help 查看说明文档
--num_img 2000000 # 生成图片数量
--corpus_mode list # 按列输出
--config_file /configs/myconfig_1.yaml # 自定义配置
--output_dir /home/XX/crnn_data/train_data # 图片输出路径
```

语料按列输出, 生成200万个样本及标签(标签集名为`tmp_labels.txt`)



## 3. lmdb数据集转换

由于crnn的训练数据集格式要求为lmdb格式,因此需要将生成的样本和标签转化.

需改动变量:

```python
# data2lmdb.py
inputPath # 元数据集读取路径
outputPath # lmdb数据集存放路径
```

数据集转换命令:

```bash
python data2lmdb.py
```



## 4. CRNN训练

使用自制的数据集训练CRNN

crnn模型: https://github.com/xiaofengShi/CHINESE-OCR.git

需改动变量:

```python
# CHINESE-OCR/train/keras-train/trainbatch.py
trainroot # 训练集读取路径
valroot # 验证集读取路径
modelPath # 预训练模型读取路径
MODEL_PATH # 再训练模型存放路径
MODEL_NAME # 再训练模型文件名
```

训练命令:

```bash
python CHINESE-OCR/train/keras-train/trainbatch.py
```



## 5. CRNN应用

使用再训练的CRNN

crnn模型: https://github.com/xiaofengShi/CHINESE-OCR.git

需改动变量:

```python
# CHINESE-OCR/ctpn/ctpn/cfg.py
CTPN_PATH # ctpn模型读取路径
# CHINESE-OCR/ocr/model.py
modelPath # keras版本的crnn模型读取路径
```

应用命令:

```bash
python CHINESE-OCR/demo.py
```

