# CycleMix: A Holistic Strategy for Medical Image Segmentation from Scribble Supervision

This project is developed for our CVPR 2022 paper: [CycleMix: A Holistic Strategy for Medical Image Segmentation from Scribble Supervision](https://arxiv.org/abs/2203.01475). Our code is implemented based on the [PuzzleMix](https://github.com/snu-mllab/PuzzleMix), but is also applicable to other mixup-based augmentation methods. For more information about CycleMix, please read the following paper:

```
@article{zhang2022cyclemix,
  title={CycleMix: A Holistic Strategy for Medical Image Segmentation from Scribble Supervision},
  author={Zhang, Ke and Zhuang, Xiahai},
  journal={arXiv preprint arXiv:2203.01475},
  year={2022}
}
```
Please also cite this paper if you are using CycleMix for your research.

# Datasets
1. The MSCMR dataset with mask annotations can be downloaded from [MSCMRseg](https://zmiclab.github.io/zxh/0/mscmrseg19/data.html) 
2. The scribble annotations of MSCMRseg have been released in [MSCMR_scribbles](https://github.com/BWGZK/CycleMix/tree/main/MSCMR_scribbles).

# Usage
1. Firstly, set the "dataset" parameter in main.py, line 76, to the name of dataset, i.e., "MSCMR_dataset". Then, set the "output_dir" in main.py, line 79, as the path to save the checkpints. Finally, set the dataset path in /data/mscmr.py, line 110, to your data path where the dataset is located in.
2. Check your GPU devices and modify the GPU parameter in main.py, line 83 and run.sh.
3. Start to train by sh run.sh.
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=5 nohup python main.py --mixup_alpha 0.5 --graph True --n_labels 3 --eta 0.2 --beta 1.2 --gamma 0.5 --neigh_size 4 --transport True --t_size 4 --t_eps 0.8 &
```

We will release the trained models as soon as possible, thanks for your attention.
