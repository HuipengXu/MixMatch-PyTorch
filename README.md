# MixMatch
This is an unofficial PyTorch implementation of [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249). 
The official Tensorflow implementation is [here](https://github.com/google-research/mixmatch).

Now only experiments on IMDB

This repository carefully implemented important details of the official implementation, However, since I have not seen 
the open source code based on the IMDB data set and experimental results, there may be problems with the results of this repository. 
If you have any questions, please raise an issue.

data augmentation methods comes from 
[EDA: Easy Data Augmentation Techniques for Boosting Performance on
 Text Classification Tasks](https://arxiv.org/pdf/1901.11196.pdf), code comes from [here](https://github.com/jasonwei20/eda_nlp)

## Requirements
- Python 3.6+
- PyTorch 1.5.1
- torchtext 0.4.0 
- tensorboardX
- progress
- matplotlib
- numpy

## Usage

### Train
Train the model by 500 labeled data of IMDB dataset:

```shell
# train baseline
python baseline.py

# preprocess data
python preprocess.py --num_labeled_examples 500

# train textcnn + mixmatch
python train.py --gpus <gpu_id> --epochs 20 --lambda-u 1.0
```
Different from the setting of the cifar data set, the setting of lambda-u on IMDB is relatively small, such as 1


## Results (Accuracy, 500 labeled data)
| lambda-u | 0.3 | 1.0 |
|:---|:---:|:---:|
|TextCNN + MixMatch | 81.30 | 81.42 |

baseline: 77.97  (25000 labeled data)  
(Results of this code were evaluated on 1 run. Results of 5 runs with different seeds will be updated later. )

## References
```
https://github.com/YU1ut/MixMatch-pytorch

@article{berthelot2019mixmatch,
  title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
  author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin},
  journal={arXiv preprint arXiv:1905.02249},
  year={2019}
}
```