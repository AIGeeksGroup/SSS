# SSS
This is the code repository for the paper:

> **SSS: Semi-Supervised SAM-2 with Efficient Prompting for Medical Imaging Segmentation**
> 
>[Hongjie Zhu](https://github.com/Potato2187/), Xiwei Liu, Rundong Xue, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)†, Yong Xu, Daji Ergu, [Ying Cai](https://ieeexplore.ieee.org/author/37087137422)* and [Yang Zhao](https://yangyangkiki.github.io/)
>
>*Corresponding author. †Project lead.
>

![项目相关的描述](image.png)

## Introduction
In the era of information explosion, efficiently leveraging large-scale unlabeled data while minimizing the reliance on high-quality pixel-level annotations remains a critical challenge in the field of medical imaging. Semi-supervised learning (SSL) enhances the utilization of unlabeled data by facilitating knowledge transfer, significantly improving the performance of fully supervised models and emerging as a highly promising research direction in medical image analysis. Inspired by the ability of Vision Foundation Models (e.g., SAM-2) to provide rich prior knowledge, we propose SSS (Semi-Supervised SAM-2), a novel approach that leverages SAM-2’s robust feature extraction capabilities to uncover latent knowledge in unlabeled medical images, thus effectively enhancing feature support for fully supervised medical image segmentation. Specifically, building upon the single-stream “weak-to-strong” consistency regularization framework, this paper introduces a Discriminative Feature Enhancement (DFE) mechanism to further explore the feature discrepancies introduced by various data augmentation strategies across multiple views. By leveraging feature similarity and dissimilarity across multi-scale augmentation techniques, the method reconstructs and models the features, thereby effectively optimizing the salient regions. Furthermore, a prompt generator is developed that integrates Physical Constraints with a Sliding Window (PCSW) mechanism to generate input prompts for unlabeled data, fulfilling SAM-2’s requirement for additional prompts. Extensive experiments demonstrate the superiority of the proposed method for semi-supervised medical image segmentation on two multi-label datasets, i.e., ACDC and BHSD. Notably, SSS achieves an average Dice score of 53.15 on BHSD, surpassing the previous state-of-the-art method by +3.65 Dice.

## Environment Setup

To set up the environment for SSS, follow these steps:

```sh
# create a clean conda environment from scratch
conda create --name SSS python=3.7
conda activate SSS

# install pip
conda install ipython
conda install pip

# install required packages
pip install -r requirements.txt
```
