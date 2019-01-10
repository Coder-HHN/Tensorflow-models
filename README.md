# Tensorflow-models
利用tensorflow实现各种model（陆续添加），同时完善TFrecord数据预处理等前期工作

Reference：
    Thanks to https://github.com/kratzert
              https://github.com/vanhuyz/CycleGAN-TensorFlow 
对两位的工作表示由衷感谢，尤其是vanhuyz的CycleGAN代码，模块划分清晰，架构合理，强烈推荐Tensorflow新手细读！

从caffe转tensorflow，一步步从零开始学习tensorflow，同时也是在将之前的一系列工作逐步开源。

dataprocess.py ---将训练数据及label转换为TFrecord。

datareader.py ----读取TFrecord，打乱样例构造batch后输出，实现了Tensorflow中标准的流水线读取过程，包括文件名队列及样例队列。

layer.py ----网络中的基本layer，注意：Tensorflow中contrib可能在新版本中被移除，因此已经尽量避免采用contrib接口

Mnet10.py ---模型文件，后续会陆续添加不同模型，设计之初就打算构造一个框架，只需替换不同model.py就可以完成不同网络

train.py ---训练文件，tensorflow基本训练过程，同时实现了存储log等功能，后续需要添加断点再训练等功能

utils.py ---辅助函数

注意：
  这个项目起初只是为了在Tensorflow上复现caffe结果，以此增添毕设工作的说服力。
  个人时间有限，重心目前在毕业相关事项，会细心校验代码，保证代码可执行，但代码中部分逻辑分支可能没有测试到，如果有bug请告诉我。
  目前代码仅实现了基础功能，后续会不断改进。
