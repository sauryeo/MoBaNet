# [IEEE TGRS 2025] MFNet

This repo is the official implementation of ['A Unified Framework With Multimodal Fine-Tuning for Remote Sensing Semantic Segmentation'](https://ieeexplore.ieee.org/abstract/document/11063320).

![framework](https://github.com/sstary/SSRS/blob/main/docs/MFNet.png)

## Usage
You can get the pre-trained model here: https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints

The core modules are in ./Model/models/ImageEncoder and ./Model/models/sam

The current mode is PEFT (`sam_peft`). You can switch between `sam_peft`, `sam_lora`, or `sam_adpt` via `-mod` in **./Model/cfg.py** (no manual code edits needed). Most of the remaining files come from the original SAM framework and can be ignored.


Run the code by: python train.py

Draw the heatmap by: python test_heatmap.py

Please cite our paper if you find it is useful for your research.

```
@ARTICLE{ma2024manet,
  author={Ma, Xianping and Zhang, Xiaokang and Pun, Man-On and Huang, Bo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={A Unified Framework With Multimodal Fine-Tuning for Remote Sensing Semantic Segmentation}, 
  year={2025},
  volume={63},
  number={},
  pages={1-15}
}
  ```
