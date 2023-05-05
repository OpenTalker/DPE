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

<br>

</div>




## 🔥 Demo

- 🔥 Video editing: single source video & a driving video & a piece of audio.
We tranfer pose through the video and transfer expression through the audio with the help of [SadTalker](https://github.com/OpenTalker/SadTalker).

| Source video   | Result  |
|:--------------------:|:--------------------: |
| <video  src="https://user-images.githubusercontent.com/34021717/235356114-cd865676-0f34-47d3-ba2d-9736e61bb3c7.mp4" type="video/mp4"> </video> | <video  src="https://user-images.githubusercontent.com/34021717/235356132-2193e346-6d89-4eb0-94a5-845d2ae5962c.mp4" type="video/mp4"> </video>  |
|  <video  src="https://user-images.githubusercontent.com/34021717/235356197-1ab8126c-bc77-4f15-90e6-eb5b49344672.mp4" type="video/mp4"> </video> | <video  src="https://user-images.githubusercontent.com/34021717/235356207-35629422-1bbe-45bb-8400-960b2bd196ed.mp4" type="video/mp4"> </video> |
| <video  src="https://user-images.githubusercontent.com/34021717/235356229-92bab207-a769-4141-8869-0db4faad41b2.mp4" type="video/mp4"> </video> | <video  src="https://user-images.githubusercontent.com/34021717/235356249-398515ac-7afa-41cd-98b2-a3ed85ed9954.mp4" type="video/mp4"> </video> |
	
- 🔥 Video editing: single source image & a driving video & a piece of audio.
We tranfer pose through the video and transfer expression through the audio with the help of [SadTalker](https://github.com/OpenTalker/SadTalker).

<video  src="https://user-images.githubusercontent.com/34021717/236397002-16a08a50-5dc7-4668-b831-c44d118d5337.mp4" type="video/mp4"> </video>
<video  src="https://user-images.githubusercontent.com/34021717/236397030-2ad811b9-bf51-4e23-bfa8-7047bc7a9a40.mp4" type="video/mp4"> </video>




- 🔥 Video editing: single source image & two driving videos.
We tranfer pose through the first video and transfer expression through the second video.
Some videos are selected from [here](https://www.colossyan.com/).

![dpe](./docs/demo1_res.gif)
![dpe](./docs/demo3_res.gif)

## 📋 Changelog

- 2023.04.30 Add some demos.
- 2023.03.18 Support `Pose driving`，`Expression driving` and `Pose and Expression driving`.
- 2023.03.18 Upload the pre-trained model, which is fine-tuning for expression generator.
- 2023.03.03 Release the test code!
- 2023.02.28 DPE has been accepted by CVPR 2023!

</details>

<!-- ## 🎼 Pipeline
![main_of_sadtalker](https://user-images.githubusercontent.com/4397546/222490596-4c8a2115-49a7-42ad-a2c3-3bb3288a5f36.png)  -->


  ## 🚧 TODO
  - [x] Test code for video driving.
  - [x] Some demos.
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

#### Crop full video
```
python crop_video.py
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
