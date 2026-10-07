# 结果文件说明

以下数值和图像由脚本或 Notebook 实际运行生成，最新运行信息见 `run_info.json`。重复运行会覆盖生成文件。

| 文件 | 内容 |
|---|---|
| metrics.json / metrics.csv | 三个模型的测试分数、正确与错误数、划分和 k |
| knn_cv.csv / knn_cv.png | 训练集内五折分数及均值、折间标准差 |
| predictions.csv | 每个模型的全部 450 张测试预测，原始编号可追溯 |
| errors.csv | predictions 中的所有错误行 |
| *_classification_report.txt | 每类 precision、recall、F1、support |
| *_confusion_matrix.png | 行是真实标签，列是预测标签 |
| *_errors.png | 测试数组顺序中的前八个错误，带原始编号 |
| environment.txt | 本次运行的四个关键包版本 |
| run_info.json | 时间、Python、划分索引、参数选择与实验函数代码指纹 |

数据来自 scikit-learn 内置 digits；CSV 中的编号是原始数据下标。没有个人身份数据或本机路径。具体解释见 [实验报告](../docs/EXPERIMENT_REPORT.md)。
