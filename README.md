# orion-eye
-----------------------------------------------------------------------------------
$$\sqrt{3x-1}+(1+x)^2$$
## brief intro
 The purpose of this project is identifying the hand-writing digits,
 Notice that this project is actually accomplished one functionality, which has
 an input image ( maybe with gray scale or RGB type ), and one output the infered
 label ( 0 - 9) or NAN( not a number happens when the input actually sends no red pixels)

-----------------------------------------------------------------------------------
## Contents
>> * 1. one model , which is pretrained in **caffe**. Here we choose **LeNet-5**, this model
* contains 9 layers in the neural net:
>>>> * (1) Data input layer, the input data should be 28 * 28 size and with pixels' value from
>>>> 0 to 255. the program will normalize this value to 0 to 1 automatically by multiply some
>>>> constant.
>>>> * (2)

the program is building forward model for calculating the
probability of input image

the original model is trained using MNIST data
