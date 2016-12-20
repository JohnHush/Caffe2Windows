# orion-eye

**Hand-writting Digits Recognition System, designed by *John Hush* @ Shanghai, Nov, 2016.**
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
...
***
