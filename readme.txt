This package is an implementation of the algorithm developed in the following paper:
Sparseness Meets Deepness: 3D Human Pose Estimation from Monocular Video.
X. Zhou, M. Zhu, S. Leonardos, K. Derpanis, K. Daniilidis. CVPR 2016.

How to use?
- first run stratup.m
- see demoH36M for an example from Human3.6M dataset
- see demoHG for an example of how to use our algorithm combined with the "Stacked hourglass network"
- see demoMPII for an example of how to reconstruct 3D poses from a single image from MPII dataset

Notes:
- the code for hourglass network in pose-hg-demo is from Newell et al., https://github.com/anewell/pose-hg-demo
- see the comments in demoHG.m for how to run hourglass network on your images and save heatmaps
- if you want to use the hourglass network, you need to first install Torch and make it work
- generally "Hourglass network" + "poseDict-all-K128" (pose dictionary learned from Human3.6M) work well 
- for better 3D reconstruction, you can learn a 3D pose dictionary using your own mocap data 
- for more details on pose dictionary learning, please see the following project:
  http://cis.upenn.edu/~xiaowz/shapeconvex.html

If you used this package in your work, please cite the following papers:
@inproceedings{zhou2016sparseness,
  title={Sparseness Meets Deepness: 3D Human Pose Estimation from Monocular Video},
  author={Zhou, Xiaowei and Zhu, Menglong and Derpanis, Kosta and Daniilidis, Kostas},
  booktitle={CVPR},
  year={2016}
}
@article{zhou2016sparse,
  title={Sparse Representation for 3D Shape Estimation: A Convex Relaxation Approach},
  author={Zhou, Xiaowei and Zhu, Menglong and Leonardos, Spyridon and Daniilidis, Kostas},
  journal={arXiv preprint arXiv:1509.04309},
  year={2016}
}

