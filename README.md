# 手写数字分类

用 scikit-learn 的 digits 数据比较 SVM 和 KNN。每张图片是 8×8 灰度图，展开成 64 个特征后进行分类。

训练集 1347 张，测试集 450 张，随机种子为 42。标准化放在 Pipeline 中，只在训练数据上拟合。KNN 的 k 在训练集内通过五折交叉验证选择，候选值为 1、3、5、7、9。

## 运行

在仓库文件夹里执行：

```bash
python -m pip install -r requirements.txt
python experiment.py
```

也可以在 VS Code 打开 `experiment.ipynb`，选择安装过依赖的 Python 内核，从上到下运行。实验不需要 GPU。

## 结果

| 模型 | 测试准确率 | 错误数 |
|---|---:|---:|
| SVM，RBF 核，C=1 | 98.00% | 9 |
| KNN，交叉验证选择 k=5 | 96.44% | 16 |

这次划分中 SVM 少错 7 张。交叉验证选出的 k 与之前的 k=5 基线一致，没有提高测试分数。以上是固定划分下的结果。

`results/metrics.json` 保存数值结果；下面是 SVM 的前八个错误，图中的 id 是原始数据编号。

![SVM 错误样本](results/svm_errors.png)

混淆矩阵见 [这里](results/svm_confusion_matrix.png)，行是真实标签，列是预测标签。

参考 [scikit-learn 官方示例](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html)，本实验的数据划分与参数设置有所不同。代码整理使用了 AI 辅助。
