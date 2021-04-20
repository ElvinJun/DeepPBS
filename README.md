# DeepPBS
**Motivation:** Accurate all-atom protein structures play an important role in various research and applications. However, in most cases, only coarse-grained models can be obtained for reasons. Precisely predict protein backbone structures based on alpha-carbon traces, the most-used coarse-grained model, is a pivotal step for precise all-atom modeling for protein structures. 

**Results:** In this study, we proposed a deep learning-based method to predict protein backbone structures from alpha-carbon traces. Our method achieved comparable performance as the best previous method with cRMSD between predicted coordinates and reference coordinates as measurement.

# Workflow
![image](https://user-images.githubusercontent.com/46809259/115357912-b3290e00-a1ef-11eb-8b82-0b58706c48a9.png)

# Webserver
[点击进入骨架结构预测网页](http://deeppbs.com/)
* Python / Pytorch / Django
* KNN / Bi-litsm / Rodrigues


# Protein structure prediction process
![](https://github.com/ElvinJun/DeepPBS/blob/master/process.jpg?raw=true)


# Protein backbone strcture prediction based on Bi-LSTM
![deeppbs](https://github.com/ElvinJun/DeepPBS/blob/master/our_process%20.jpg?raw=true)


# Method of rotation repetition
![rotation](https://github.com/ElvinJun/DeepPBS/blob/master/rotation.jpg?raw=true)
