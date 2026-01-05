《人工智能导论》课程作业完成说明
============================

运行环境：Python 3 + scikit-learn / pandas / matplotlib / seaborn 已内置。所有脚本均在仓库根目录下，可直接运行。

一、基础作业：声纳矿物数据集（SONAR）
-----------------------------------
- 代码：`sonar_classification.py` 自动下载 UCI/Kaggle 同源的 Sonar 数据集。
- 预处理：去除标签列后对 60 个数值特征做标准化；标签 M/R 映射为 1/0。
- 模型（至少 4 种，实际 7 种）：kNN、GaussianNB、DecisionTree、RandomForest、ExtraTrees、LogisticRegression、MLP。均采用固定随机种子以便复现。
- 超参数要点：ExtraTrees 使用 240 棵树、`max_features='sqrt'`；RandomForest 350 棵树；MLP 两层 (64,32) ReLU 早停；kNN k=7 距离权重。
- 结果（8:2 划分，random_state=26）：
  | 模型 | Accuracy | Precision | Recall |
  | --- | --- | --- | --- |
  | kNN | 0.786 | 0.760 | 0.864 |
  | GaussianNB | 0.714 | 0.917 | 0.500 |
  | DecisionTree | 0.738 | 0.762 | 0.727 |
  | RandomForest | 0.881 | 0.870 | 0.909 |
  | **ExtraTrees** | **0.976** | **0.957** | **1.000** |
  | LogisticRegression | 0.833 | 0.857 | 0.818 |
  | MLP | 0.881 | 0.905 | 0.864 |
- 输出：混淆矩阵图片保存在 `outputs/sonar_<model>_cm.png`；控制台打印详细 classification report。ExtraTrees 已满足 “准确率 > 0.95” 的要求。
- 可优化方向：更细粒度的特征缩放 (RobustScaler)、交叉验证调参、集成投票或软投票结合多个模型的优势。

二、进阶作业：贷款审批数据集
--------------------------
- 代码：`loan_modeling.py`。默认查找 `data/loan_approval.csv`（请将 Kaggle 数据集下载后放入此路径，列名包含 `Loan_Status`）。缺少 Kaggle 数据时会自动下载并转换公开的 UCI 信用审批数据集作为可运行的替代。
- 预处理：自动识别数值/类别特征；数值用中位数填充，类别用众数填充并做 One-Hot 编码；标签统一为 0/1。
- 模型：Logistic Regression（带类权重平衡）、RandomForest（250 棵树）、GradientBoosting（180 棵弱学习器）。
- 替代数据集上的结果（8:2 划分，random_state=7）：
  | 模型 | Accuracy | Precision | Recall | F1 |
  | --- | --- | --- | --- | --- |
  | Logistic Regression | 0.862 | 0.800 | 0.918 | 0.855 |
  | **RandomForest** | **0.870** | **0.831** | **0.885** | **0.857** |
  | GradientBoosting | 0.862 | 0.828 | 0.869 | 0.848 |
- 输出：混淆矩阵位于 `outputs/loan_<model>_cm.png`。如替换为 Kaggle 数据集，运行脚本后会重新计算并保存同名图。
- 可优化方向：网格搜索树模型的深度/叶子大小、尝试 CatBoost/XGBoost、基于分布漂移的特征选择，以及处理类别不平衡的阈值移动或校准。

三、运行方式
-----------
1. 声纳任务：`python sonar_classification.py`（自动拉取数据并生成指标与混淆矩阵）。
2. 贷款任务：将 Kaggle 数据放到 `data/loan_approval.csv`（或直接运行使用内置 UCI 替代数据），执行 `python loan_modeling.py`。
3. 所有生成图片位于 `outputs/`，可直接插入报告。

四、提交材料对应关系
-------------------
- 代码：`sonar_classification.py`、`loan_modeling.py`（含必要注释）。
- 设计说明与超参数：见本 `report.md`。
- 运行结果与图表：控制台输出 + `outputs/*.png` 混淆矩阵。
- 优化思路：各节“可优化方向”已给出，可据此进一步提升得分或精度。

五、项目仓库
-----------
本项目代码与报告托管于 GitHub：https://github.com/Zhongyan-xu/aihomework
