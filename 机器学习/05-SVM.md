# 支持向量机SVM

支持向量机主要用于线性不可分问题的解决，用于解决分类和回归问题。SVM通过找到**数据点之间的最优边界**来区分不同的类别。在SVM中，**核函数**起着至关重要的作用，允许SVM在高维空间中有效地处理非线性问题

**将数据投射至高维空间**。这正是**SVM算法的核函数（kernel trick）功能**，在SVM中用得最普遍的两种把数据投射到高维空间的方法分别是**多项式内核（Polynomial kernel）和径向基内核（Radial basis function kernel, RBF）**。其中多项式内核比较容易理解，它是通过把样本原始特征进行乘方来把数据投射到高维空间，比如特征1乘2次方、特征2乘3次方，特征3乘5次方等。而RBF内核也被称为高斯内核（Gaussian kernel）

![image-20240621150945224](https://51catgithubio.oss-cn-beijing.aliyuncs.com/image-20240621150945224.png)

**在SVM算法中，训练模型的过程实际上是对每个数据点对于数据分类决定边界的重要性进行判断。也就是说，在训练数据集中，只有一部分数据对于边界的确定是有帮助的，而这些数据点就是正好位于决定边界上的。这些数据被称为“支持向量”（support vectors）**

SVM中通过支持向量来选择分隔超平面，分隔超平面将训练样本分为正反两派，支持向量的作用就是使得选择的分隔超平面离两边的类别都**比较远**

## 例子

线性内核的支持向量模型

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=50, centers=2, random_state=6)
clf = svm.SVC(kernel="linear", C = 1000)
clf.fit(X,y)

xx, yy = np.linspace(x_lim[0], x_lim[1], 30), np.linspace(y_lim[0], y_lim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
# 绘图
plt.scatter(X[:,0], X[:,1], c = y)
ax = plt.gca()
x_lim=ax.get_xlim()
y_lim=ax.get_ylim()
ax.contour(XX, YY, Z, colors = 'k', levels = [-1,0,1],alpha = 0.5, linestyles = ['--', '-', '--'])
ax.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], c = 'red')
plt.show()
```

![image-20240621155428663](https://51catgithubio.oss-cn-beijing.aliyuncs.com/image-20240621155428663.png)

 `clf.decision_function`: 是一个用于评估样本与决策边界关系的函数。它计算了样本与支持向量之间的距离，或者在非线性SVM中，计算了核函数在样本上的应用结果，计算样本点到分割超平面的**函数距离**？ 不太明白

`clf.support_vectors_`: 支持向量的点

红色点就是支持向量，本例使用的这种方法称为**“最大边界间隔超平面”（Maximum Margin Separating Hyperplane）**。指的是说中间这条实线（在高维数据中是一个超平面），和所有支持向量之间的距离，都是最大的

## 核函数与参数

不同核函数的选择对分类的影响

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC, LinearSVC

# 导入红酒数据集
wine = datasets.load_wine()

# 定义一个函数用来画图
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

# 定义一个绘制等高线的函数
def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# 选取数据集的前两个特征
X = wine.data[:, :2]
y = wine.target

# SVM 的正则化参数
C = 1.0
models = (SVC(kernel='linear', C=C),
          LinearSVC(C=C),
          SVC(kernel='rbf', gamma=0.7, C=C),
          SVC(kernel='poly', degree=3, C=C))

models = (clf.fit(X, y) for clf in models)

# 设定图题
titles = ('svc with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# 设定一个子图形的个数和排列方式
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# 使用前面定义的函数进行画图
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)
for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.plasma, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.plasma, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

# 将图形显示出来
plt.show()
```



![image-20240621161823828](https://51catgithubio.oss-cn-beijing.aliyuncs.com/image-20240621161823828.png)

线性内核的SVC与linearSVC得到的结果非常近似，但仍然有一点点差别。其中一个原因是linearSVC对L2范数进行最小化，而线性内核的SVC是对L1范数进行最小化。不论如何，linearSVC和线性内核的SVC生成的决定边界都是线性的，**在更高维数据集中将会是相交的超平面**

而RBF内核的SVC和polynomial内核的SVC分类器的决定边界则完全不是线性的，它们更加弹性。而决定了它们决定边界形状的，就是它们的参数。

- polynomial内核的SVC中，起决定性作用的参数就是**degree和正则化参数*C***，在本例中我们使用的degree为3，也就是对原始数据集的特征进行乘3次方操作。

- RBF内核的SVC中，起决定作用的是**正则化参数*C*和参数gamma**

### 不同gamma值对SVM的影响

<img src="https://51catgithubio.oss-cn-beijing.aliyuncs.com/image-20240621162313502.png" alt="image-20240621162313502" style="zoom:80%;" />

gamma值越小，则RBF内核的直径越大，这样就会有更多的点被模型圈进决定边界中，所以决定边界也就越平滑，这时的模型也就越简单；而随着参数的增加，模型则更倾向于把每一个点都放到相应的决定边界中，这时模型的复杂度也相应提高了。所以**gamma值越小，模型越倾向于欠拟合，而gamma值越大，则模型越倾向于出现过拟合的问题。**

在支持向量机（SVM）中的 `C` 参数和岭回归（Ridge Regression）/ lasso回归中的 `alpha` 参数都用于控制模型的正则化强度，但它们在模型中扮演的角色和作用机制是不同的：

1. **SVM中的 `C` 参数**：
   - `C` 是 SVM 的正则化参数，它控制着错误项（违反间隔的样本点）的惩罚力度。在 SVM 中，**较大的 `C` 值会增加对误分类样本的惩罚，使模型更关注于正确分类所有样本，可能导致过拟合**。**较小的 `C` 值会减少对误分类样本的惩罚，允许更多的样本违反间隔，从而增加间隔宽度，可能导致欠拟合。**
2. **岭回归中的 `alpha` 参数**：
   - `alpha` 是岭回归中的正则化参数，它是 L2 范数正则化的系数。在岭回归中，`alpha` 值用于控制模型系数的平滑程度，**较大的 `alpha` 值会导致模型系数更小，从而减少模型复杂度，避免过拟合**。**较小的 `alpha` 值可能使模型系数较大，模型可能过于复杂，容易过拟合**。
3. **正则化的目的**：
   - 无论是 `C` 还是 `alpha`，它们都用于平衡模型的拟合度和复杂度，防止模型在训练数据上过拟合。
4. **作用机制**：
   - 在 SVM 中，正则化是通过惩罚违反间隔的错误项来实现的，而岭回归是通过在损失函数中添加一个与系数平方成正比的惩罚项来实现的。

### SVM的缺点和优势

如果数据集中的样本数量在1万以内，SVM都能驾驭得了，但如果样本数量超过10万的话，SVM就会非常耗费时间和内存，SVM还有一个短板，因为要计算距离，特征的大小和数量级影响是比较大的，所以对于数据预处理和参数调节要求高，假设数据集中样本特征的测度都比较接近，例如在图像识别领域，还有样本特征数和样本数比较接近的时候，SVM都会游刃有余，比如之前的扩增子测序

在SVM算法中，有3个参数是比较重要的：第一个是核函数的选择；第二个是核函数的参数，例如RBF的gamma值；第三个是正则化参数*C*。RBF内核的gamma值是用来调节内核宽度的，**gamma值和*C*值一起控制模型的复杂度，数值越大模型越复杂，而数值越小模型越简单**。实际应用中，gamma值和*C*值往往要一起调节，才能达到最好的效果

## 实例

波士顿房价数据集https://zhuanlan.zhihu.com/p/626831299被政治正确移除了（狗头

在sklearn1.2后移除了所以按照官方提供的另一个方法下载

<img src="https://51catgithubio.oss-cn-beijing.aliyuncs.com/image-20240621171801452.png" alt="image-20240621171801452" style="zoom:80%;" />

```python
import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
```

做一个简单的例子

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, target)
for k in ['linear', 'rbf']:
    clf = SVR(kernel=k)
    clf.fit(X_train, y_train)
    print(f"{k} 测试集: {clf.score(X_test, y_test)}")
    print(f"{k} 训练集: {clf.score(X_train, y_train)}")    
# out: rbf核函数的训练结果特别差
'''
linear 测试集: 0.6533952831660281
linear 训练集: 0.7351569455230236
rbf 测试集: 0.2300384367456073
rbf 训练集: 0.19614389454527292
'''

# 对数据做标准化 因为要计算距离
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # z-score
scaler.fit(X_train)
X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)

# 调节gammar和C参数 改善明显
for k in ['linear', 'rbf']:
    clf = SVR(kernel=k, C = 300, gamma=0.01)
    clf.fit(X_train_scale, y_train)
    print(f"{k} 测试集: {clf.score(X_test_scale, y_test)}")
    print(f"{k} 训练集: {clf.score(X_train_scale, y_train)}")  
'''
linear 测试集: 0.6663599777573498
linear 训练集: 0.735017322000028
rbf 测试集: 0.8204826669434937
rbf 训练集: 0.9126739838690716
'''
```

