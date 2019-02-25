# semantic segmentation
## FCN
* [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
* FCN论文解析

  这篇论文是semantic segmentation领域比较基础的论文，在次论文的基础上发展出了一系列更先进的方法，
  但是要理解semantic segmentation还是需要从这篇论文开始阅读。
  
  论文主要的创新有两点：
  
  1.设计了一种全卷积网络，将FC连接用卷积来实现。
  
  2.通过像素级别的比较来训练FCN网络

* FCN32

  ***网络结构***
  
  ![FCN32](https://github.com/stesha2016/tensorflow-semantic-segmentation/blob/master/images/FCN32.png)
  
  经过了5次pool，所以图片缩小了32倍，然后在基于这个尺寸的图片恢复到了原图尺寸，所以叫做FCN32
  
  ***数据准备***
  
  使用[LabelImgTool
](https://github.com/lzx1413/LabelImgTool)之类的工具对图片进行标记，标记后会生成一张新的图片，
这张图片在你标记的物体1的区域的像素点都是(1,1,1)，在物体2的区域像素点都是(2,2,2)，以此类推。
  
  ***Loss约束***
  
  从网络结构可以看出最后网络的output是224x224xclasses的矩阵，而ground truth就是我们标记出来的数据是224x224x3，
  我们需要对ground truth做一点改变。对224x224x3的ground truth取出一个色度224x224，那么这个224x224的矩阵中物体1
  的位置的数值是1，物体2的位置的数值为2，背景位置的数字为0，然后我们用一种类似one hot的方式就可以将数据变成224x224xclasses。
  
  比如如果classes为5，那么ground truth矩阵中，物体1的位置数组是[0, 1, 0, 0, 0]，物体2的位置数组是[0, 0, 1, 0, 0],
  背景位置的数组是[1, 0, 0, 0, 0]，这样就可以顺利的和网络结构生成的矩阵在axis=2的维度计算cross entropy loss了。
  
* FCN16
  
  ***网络结构***
  
  ![FCN16](https://github.com/stesha2016/tensorflow-semantic-segmentation/blob/master/images/FCN16.png)
  
  经过了5次pool后，与pool4相加后恢复成原图的尺寸，pool4相对原图缩小了16倍，所以叫做FCN16
  
  ***数据准备***
  
  同FCN32
  
  ***Loss约束***
  
  同FCN32
  
* FCN8
  
  ***网络结构***
  
  ![FCN8](https://github.com/stesha2016/tensorflow-semantic-segmentation/blob/master/images/FCN8.png)
  
  经过了5次pool后，将pool5与pool4都恢复到pool3的尺寸，然后与pool3相加后的结果再恢复成原图尺寸，pool3相对原图缩小了8倍，所以叫做FCN8
  
  ***数据准备***
  
  同FCN32
  
  ***Loss约束***
  
  同FCN32
  
## UNET
[论文地址](https://arxiv.org/pdf/1505.04597.pdf)

***网络结构***

结构图摘自论文

![FCN8](https://github.com/stesha2016/tensorflow-semantic-segmentation/blob/master/images/UNET.png)

UNET其实就是FCN的改进版本，FCN的decode过程是直接放大，这样很多信息都无法还原，UNET就采用逐步放大图片，并且与对应的encode层
相加来保留图片的一些信息。

但是论文中是逐步缩小了图片，论文中提到是处于内存考虑，我这边代码实现不会对图片尺寸做缩小。