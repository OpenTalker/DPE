<div align="center">

<h2> DPE： <span style="font-size:12px">Disentanglement of Pose and Expression for General Video Portrait Editing </span> </h2> 

  <a href='https://arxiv.org/abs/2301.06281'><img src='https://img.shields.io/badge/ArXiv-2211.14758-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='https://carlyx.github.io/DPE/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() 

<div>
    <a href='https://carlyx.github.io/' target='_blank'>Youxin Pang <sup>1,2,3</sup> </a>&emsp;
    <a href='https://yzhang2016.github.io/' target='_blank'>Yong Zhang <sup>3,*</sup></a>&emsp;
    <a href='https://weizequan.github.io/' target='_blank'>Weize Quan <sup>1,2</sup></a>&emsp;
    <a href='https://sites.google.com/site/yanbofan0124/' target='_blank'>Yanbo Fan <sup>3</sup></a>&emsp;
    <a href='https://vinthony.github.io/' target='_blank'>Xiaodong Cun <sup>3</a>&emsp;
    <a href='https://scholar.google.com/citations?hl=zh-CN&user=4oXBp9UAAAAJ' target='_blank'>Ying Shan <sup>3</sup> </a>&emsp;
    <a href='https://sites.google.com/site/yandongming/' target='_blank'>Dong-ming Yan <sup>1,2,*</sup> </a>&emsp;
</div>
<br>
<div>
    <sup>1</sup> MAIS & NLPR, Institute of Automation, Chinese Academy of Sciences, Beijing, China &emsp; <sup>2</sup> School of Artificial Intelligence, University of Chinese Academy of Sciences &emsp; <sup>3</sup> Tencent AI Lab, ShenZhen, China &emsp; 
</div>
<br>
<i><strong><a href='https://arxiv.org/abs/2301.06281' target='_blank'>CVPR 2023</a></strong></i>
<br>
<br>

<!-- ![sadtalker](https://user-images.githubusercontent.com/4397546/222490039-b1f6156b-bf00-405b-9fda-0c9a9156f991.gif)

<b>TL;DR: A realistic and stylized talking head video generation method from a single image and audio.</b> -->

<br>

</div>


## 📋 Changelog

- 2023.03.18 Support `Pose driving`，`Expression driving` and `Pose and Expression driving`.
- 2023.03.18 Upload the pre-trained model, which is fine-tuning for expression generator.
- 2023.03.03 Release the test code!
- 2023.02.28 DPE has been accepted by CVPR 2023!

</details>

<!-- ## 🎼 Pipeline
![main_of_sadtalker](https://user-images.githubusercontent.com/4397546/222490596-4c8a2115-49a7-42ad-a2c3-3bb3288a5f36.png)  -->


## 🚧 TODO
- [x] Test code for video driving.
- [ ] Gradio/Colab Demo.
- [ ] Training code of each componments.
- [ ] Test code for video editing.
- [ ] Integrate audio driven methods for video editing.
- [ ] Integrate [GFPGAN](https://github.com/TencentARC/GFPGAN) for face enhancement.


## 🔮 Inference

#### Dependence Installation

<details><summary>CLICK ME</summary>

```
git clone https://github.com/Carlyx/DPE
cd DPE 
conda create -n dpe python=3.8
source activate dpe
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt

### install gpfgan for enhancer
pip install git+https://github.com/TencentARC/GFPGAN

```  

</details>

#### Trained Models
<details><summary>CLICK ME</summary>

Please download our [pre-trained model](https://drive.google.com/file/d/18Bi06ewhcx-1owlJF3F_J3INlXkQ3oX2/view?usp=share_link) and put it in ./checkpoints.



| Model | Description
| :--- | :----------
|checkpoints/dpe.pt | Pre-trained model (V1).

</details>

#### Expression driving
```
python run_demo.py --s_path ./data/s.mp4 \
 		--d_path ./data/d.mp4 \
		--model_path ./checkpoints/dpe.pt \
		--face exp \
		--output_folder ./res
```

#### Pose driving
```
python run_demo.py --s_path ./data/s.mp4 \
 		--d_path ./data/d.mp4 \
		--model_path ./checkpoints/dpe.pt \
		--face pose \
		--output_folder ./res
```

#### Expression and pose driving
```
python run_demo.py --s_path ./data/s.mp4 \
 		--d_path ./data/d.mp4 \
		--model_path ./checkpoints/dpe.pt \
		--face both \
		--output_folder ./res
```

#### Video editing
```
  TODO
```

#### Video editing for audio driving
```
  TODO
```

## 🛎 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{pang2023dpe,
  title={DPE: Disentanglement of Pose and Expression for General Video Portrait Editing},
  author={Pang, Youxin and Zhang, Yong and Quan, Weize and Fan, Yanbo and Cun, Xiaodong and Shan, Ying and Yan, Dong-ming},
  journal={arXiv preprint arXiv:2301.06281},
  year={2023}
}
```

## 💗 Acknowledgements
Part of the code is adapted from 
[LIA](https://github.com/wyhsirius/LIA),
[PIRenderer](https://github.com/RenYurui/PIRender),
[STIT](https://github.com/rotemtzaban/STIT).
We thank authors for their contribution to the community.


## 🥂 Related Works
- [StyleHEAT: One-Shot High-Resolution Editable Talking Face Generation via Pre-trained StyleGAN (ECCV 2022)](https://github.com/FeiiYin/StyleHEAT)
- [CodeTalker: Speech-Driven 3D Facial Animation with Discrete Motion Prior (CVPR 2023)](https://github.com/Doubiiu/CodeTalker)
- [VideoReTalking: Audio-based Lip Synchronization for Talking Head Video Editing In the Wild (SIGGRAPH Asia 2022)](https://github.com/vinthony/video-retalking)
- [SadTalker： Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023)](https://github.com/Winfredy/SadTalker)
- [3D GAN Inversion with Facial Symmetry Prior (CVPR 2023)](https://github.com/FeiiYin/SPI/)
- [T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations (CVPR 2023)](https://github.com/Mael-zys/T2M-GPT)

## 📢 Disclaimer

This is not an official product of Tencent. This repository can only be used for personal/research/non-commercial purposes.
