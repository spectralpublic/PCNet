# Pairwise Comparison Network for Remote Sensing Scene Classification

### 1. Introduction

This is the reserch code of the IEEE Geoscience and Remote Sensing Letters 2022 paper.

Y. Zhang, X. Zheng, and X. Lu, “Pairwise Comparison Network for Remote Sensing Scene Classification,” IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1-5, 2022.

In this code, we selected pairs of confused images as input and generate their own self- representation and mutual-representation. The self-representation highlights the informative parts of each image itself and the mutual-representation captures the subtle differences between image pairs. While a ranking loss is introduced to consider the feature priorities: the self-representation should be more discriminative than the mutual representation.


### 2. Start

Requirements:
             
             Python 2.7
 
             Pytorch 0.4.1
 
             torchvision 0.2.0

1. Build train.txt and test.txt list 

2. Run "python train.py" 



### 3. Related work

If you find the code and dataset useful in your research, please consider citing:

    @article{zhang2021pairwise,
     title={Pairwise Comparison Network for Remote Sensing Scene Classification},
     author={Zhang, Yue and Zheng, Xiangtao and Lu, Xiaoqiang},
     journal={IEEE Geoscience and Remote Sensing Letters},
     year={2021},
     publisher={IEEE}
     }

     @inproceedings{zhuang2020learning,
      title={Learning attentive pairwise interaction for fine-grained classification},
      author={Zhuang, Peiqin and Wang, Yali and Qiao, Yu},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      volume={34},
      number={07},
      pages={13130--13137},
      year={2020}
      }

      @article{chen2022remote,
       title={Remote Sensing Scene Classification by Local--Global Mutual Learning},
       author={Chen, Xiumei and Zheng, Xiangtao and Zhang, Yue and Lu, Xiaoqiang},
       journal={IEEE Geoscience and Remote Sensing Letters},
       volume={19},
       pages={1-5},
       year={2022},
       publisher={IEEE}
       }




