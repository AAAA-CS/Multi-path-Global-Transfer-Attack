# Multi-path-Global-Transfer-Attack
# Prerequisites
Python 3.8.16<br>
Pytorch 1.12.1<br>
Numpy 1.22.4<br>
Scipy 1.7.3<br>

# Usage
# Usage
1. To train the "train.py" with dataset PvaiaU ,which will generate checkpoint:'/SwinTransformer.pkl'.  It's trained by a simple Swin classifier,you can try other targetmodel，such as 3D-CNN, 3D-DL.<br>
 ```asp
                        $ python train.py --dataset PaviaU --train 
   ```  

2. Run the "test.py" to generate adversarial examples.<br>
  ```asp
                             $ python "test.py" --dataset PaviaU
   ```
					  
# Related works
>[ Hyperspectral-Classification](https://github.com/eecn/Hyperspectral-Classification)"> Hyperspectral-Classification")<br>
[ Swin-Transformer](https://github.com/microsoft/Swin-Transformer)"> Swin-Transformer")
