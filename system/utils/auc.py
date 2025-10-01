import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 创建一个二分类问题的数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用逻辑回归作为分类器
clf = LogisticRegression(solver='liblinear', random_state=42)
clf.fit(X_train, y_train)

# 预测概率
y_score = clf.predict_proba(X_test)[:, 1]

# 计算ROC曲线的真正类率和假正类率
fpr, tpr, thresholds = roc_curve(y_test, y_score)

# 计算AUC值
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC 曲线 (面积 = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正例率',fontsize=14)
plt.ylabel('真正例率',fontsize=14)
plt.title('ROC曲线',fontsize=14)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('AUC.png')
plt.show()