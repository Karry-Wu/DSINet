# DSINet

## Data Preparation
All datasets used can be downloaded at [here](https://pan.baidu.com/s/14B9-3j686Kd9ejz92TdpDg ) [e8bt]. 

### Training set
We use the training set of [DUTS](http://saliencydetection.net/duts/) to train our DSINet. 

### Testing Set
We use the testing set of [DUTS](http://saliencydetection.net/duts/), [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html), [PASCAL-S](http://cbi.gatech.edu/salobj/), [DUT-O](http://saliencydetection.net/dut-omron/), to test our DSINet. After downloading, put them into `/datasets` folder.

Your `/datasets` folder should look like this:

````
-- datasets
   |-- DUT-O
   |   |--imgs
   |   |--gt
   |-- DUTS-TR
   |   |--imgs
   |   |--gt
   |-- ECSSD
   |   |--imgs
   |   |--gt
   ...
````

## Training and Testing
1. Download the pretrained backbone weights and put it into `backboneweight/` folder. [HRNet] (https://pan.baidu.com/s/1DRPvNPr9QPuJJccNf_uJIw)[baidu:h436]  [SwinTransformer](https://github.com/microsoft/Swin-Transformer) . 

2. Run `python train.py ` for training and testing. The prediction saliency maps and evaluation results will be in `predresult/` folder and the training records and model weights will be in `ckpt/` folder. 
