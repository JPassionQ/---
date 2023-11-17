## chapter5 神经网络

### 神经元模型

#### M-P神经元模型

* 该模型接收来自n个其他神经元传递过来的输入信号
* 输入信号通过带权重的连接进行传递
* 神经元接收到的总输入值将于神经元的阈值进行比较
* 通过**激活函数**处理以产生神经元的输出

<img src="C:\Users\86156\AppData\Roaming\Typora\typora-user-images\image-20231110140758236.png" alt="image-20231110140758236" style="zoom: 50%;" />

#### 激活函数

**理想：阶跃函数**

数学性质不好：不连续、不光滑

**sigmoid函数**：把可能在较大范围内变化的输入值挤压到（0，1）输出值范围内

![image-20231110141300011](C:\Users\86156\AppData\Roaming\Typora\typora-user-images\image-20231110141300011.png)

### 感知机与多层网络

#### 感知机

* 两层神经元组成
* 输入层接收外界输入信号传递给输出层
* 输出层使M-P神经元：**阈值逻辑单元**
* 能实现与、或、非运算

<img src="C:\Users\86156\AppData\Roaming\Typora\typora-user-images\image-20231110141658583.png" alt="image-20231110141658583" style="zoom: 25%;" />

阈值和权重的学习可统一为权重的学习：
$$
\omega_i \gets \omega_i+\Delta \omega_i\\
\Delta\omega_i=\eta(y-y_i)x_i
$$
感知机只能解决两类模式的线性分类问题，不能解决**异或**这样的非线性可分问题

#### 多层网络

* 加入隐含层，隐含层和输出层神经元都是拥有激活函数的功能神经元
* 每层神经元与下一层神经元全互联，神经元之间不存在同层连接，也不存在跨层连接
* 多层前馈神经网络
* 输入层神经元仅是接收输入，不进行函数处理
* 隐层与输出层包含功能神经元

<img src="C:\Users\86156\AppData\Roaming\Typora\typora-user-images\image-20231110150507501.png" alt="image-20231110150507501" style="zoom:33%;" />

<img src="C:\Users\86156\AppData\Roaming\Typora\typora-user-images\image-20231110150523091.png" alt="image-20231110150523091" style="zoom:33%;" />

### 误差逆传播

**BP网络**一般是指用BP算法训练的多层前馈神经网络

<img src="C:\Users\86156\AppData\Roaming\Typora\typora-user-images\image-20231110150840876.png" alt="image-20231110150840876" style="zoom:33%;" />

基本原理：求导链式法则、反向传播、均方误差

**解决过拟合**

* 早停
* 正则化

### 全局最小与局部极小

<img src="C:\Users\86156\AppData\Roaming\Typora\typora-user-images\image-20231110151159125.png" alt="image-20231110151159125" style="zoom:33%;" />

如何跳出局部极小？（启发式，可解释性不足）

* 以多组不同参数值初始化多个神经网络
* 使用**模拟退火**，每一步都以一定概率接收比当前解更差的结果
* 使用随机梯度下降
* 遗传算法

### 其他常见神经网络

* RBF网络
* ART网络
* SOM网络
* 级联相关网络
* Elman网络
* Boltzmann网络

### 深度学习

pass