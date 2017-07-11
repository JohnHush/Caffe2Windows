# orion-eye

**Hand-writting Digits Recognition System, designed by *John Hush* @ Shanghai, Nov, 2016.**

## WINDOWS下目录结构
* 生成两个lib，一个OCR_PREDICT，一个是OCR_TRAIN，其中OCR_PREDICT是在Visual Studio 2010中生成的，并且没有调用Caffe的库，只调用了OpenCV 2.4.6， OpenBLAS和protobuf，当然这三个依赖库是MSVC2010的版本；另一个lib，OCR_TRAIN是在Visual Studio 2015 上编译的，同时依赖Caffe库；
* 整个项目还要包括若干个测试程序，它们都放在./test文件夹中，还有若干个工具程序，它们都放在./tools
* 整个项目准备支持Window, Linux和MacOS三个系统，主要在文件读写方面会有一些出入

## 若干MSVC下操蛋的事
在使用Microsoft Visual Studio C++时要特别注意MSVC的版本，我们现在使用的是VS2010和VS2015。
* 使用openblas时，可能会报MSB4030错误，提示LNK GenerateDebugInformation参数是无效值。解决方案是： “属性——配置属性——链接器——调试”中 “生成调试信息”选择“否”！
* 使用openblas时，VS2010 fatal error LNK1123: 转换到 COFF 期间失败；解决方案是：将 项目|项目属性|配置属性|连接器|清单文件|嵌入清单 “是”改为“否”！
* 在使用openblas时，在release模式下可能出现出现"0x00905a4d 处未处理的异常，但是在debug模式下并没有，解决方案是： 在release版本下面使用"保留未引用数据(/OPT:NOREF)"选项！ 链接器->优化->引用   中
##  Overview

The purpose of this project is to automatically identify the hand-writing digits, Notice that this project only does one single thing! 

* **input:** an **IplImage \*** image ( could be gray scale or RGB type ).
* **output:** an output show the infered label ( 0 - 9) or NAN( when the input actually sends no red pixels )

## feedback
* 1. there will be some numerical unstable problem when using gray scale image,i.e 8.jpg
and return the determinant is zero warning..
## Contents

We basically break the project into several parts, the two main parts are **Model Training Phase** and **Data Prediction Phase**.  

* **Model Training Phase:** this is accomplished in **caffe**, with training data downloaded from [MNIST](http://yann.lecun.com/exdb/mnist/), the data contains 60,000 labeled data, with nearly the same amount of each category, say about 10,000. The Neural Network Architecture we used is CNN, specially LeNet-5.

* **Data Prediction Phase:** once we got the model, we cannot simply send our image into the *BLACK BOX*, we need to preProcess it. It almost contains the following steps:  
	- **image binarization:** the image scanned from devs contains lots of noise which will severely influent our identification, so meaningless pixels should be filtered out. In our case of identifying the digits written by teachers, the meanful part is the lines ( *black containing box lines and red score mark* )  

		* **cvAdaptiveThreshold** adaptively compute the threshold in binarization task, it could handle the lightning unevenly problem. there maybe have some random noise.
		* **Hough Transformation** is under consideration because there will be one bounding box in the image.
		* **Template Matching** also should work.
	
	- **Red Pixels Extraction:** Once we get the binarized image, we need to extract the red pixels if it has any. The color of original image is not pure which is a common phenommenon, the black pixels are not practically black, and the red pixels may not have so high brightness.  

		* A **Machine Learning** method was used to cluster the points in color space. We construct two features in this task: $$(R+\epsilon)/(G+\epsilon)$$ and $$(R+\epsilon)/(B+\epsilon)$$, while R, G, B is the red, green, blue component of one pixel, respectively. Then a **mixed Gaussian Model** is built to describe the feature data, after derived from **EM** algorithm, we could get the model parameters. Finally, we use this model to predict whether the existing pixel gets black color or red color.  

	- **Image Scaling:** at last the image should be scaled to 28\*28 size, here we resize the image to 20\*20 size first. After that, we move the image into a 28\*28 size box with its heart in the center. Be caution the image in such small size could has serious **alias** problem, this could be handled by an anti-alias algorithm.  

	- **Prediction:** if in **Red Pixels Extraction** we got a **true** return value, which means we have found some red pixels, the extracted 28\*28 size image is sent to the model, the output label is the one with highest score.  

## Concluding Remarks
 test user.name 
***
