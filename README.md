# General Video Portrait Editing

DPE: Disentanglement of Pose and Expression for General Video Portrait Editing

*Youxin Pang, Yong Zhang, Weize Quan, Yanbo Fan, Xiaodong Cun, Ying Shan, Dong-ming Yan*  
[[Paper](https://arxiv.org/abs/2301.06281)]
[[Project Page](https://carlyx.github.io/DPE/)]


## Abstract
One-shot video-driven talking face generation aims at producing a synthetic talking video by transferring the facial motion from a video to an arbitrary portrait image. Head pose and facial expression are always entangled in facial motion and transferred simultaneously. However, the entanglement sets up a barrier for these methods to be used in video portrait editing directly, where it may require to modify the expression only while maintaining the pose unchanged. One challenge of decoupling pose and expression is the lack of paired data, such as the same pose but different expressions. Only a few methods attempt to tackle this challenge with the feat of 3D Morphable Models (3DMMs) for explicit disentanglement. But 3DMMs are not accurate enough to capture facial details due to the limited number of Blendshapes, which has side effects on motion transfer. In this paper, we introduce a novel self-supervised disentanglement framework to decouple pose and expression without 3DMMs and paired data, which consists of a motion editing module, a pose generator, and an expression generator. The editing module projects faces into a latent space where pose motion and expression motion can be disentangled, and the pose or expression transfer can be performed in the latent space conveniently via addition. The two generators render the modified latent codes to images, respectively. Moreover, to guarantee the disentanglement, we propose a bidirectional cyclic training strategy with well-designed constraints. Evaluations demonstrate our method can control pose or expression independently and be used for general video editing.

## Environment
```
pip install -r requirements
```

## Inference
### Pretrained Models
Please download our [pre-trained model](https://drive.google.com/file/d/1MO6XPlkqdH7_PGHpG8a6EtvT813ktzkj/view?usp=share_link) and put it in ./checkpoints.

### Expression driving
```
python run_demo.py --s_path ./data/s.mp4 \
 		--d_path ./data/d.mp4 \
		--model_path ./checkpoints/dpe.pt \
		--face exp \
		--output_folder ./res
```
  
### Pose driving
```
python run_demo.py --s_path ./data/s.mp4 \
 		--d_path ./data/d.mp4 \
		--model_path ./checkpoints/dpe.pt \
		--face pose \
		--output_folder ./res
```

### Expression and pose driving
```
python run_demo.py --s_path ./data/s.mp4 \
 		--d_path ./data/d.mp4 \
		--model_path ./checkpoints/dpe.pt \
		--face both \
		--output_folder ./res
```

### Video editing(Coming soon)
```
python run_demo_paste.py 
 --input_folder /path/to/images_dir \
 --output_folder /path/to/experiment_dir \
 --face exp \
```

## **Citation**

If you find our work useful in your research, please consider citing:

```
@article{pang2023dpe,
  title={DPE: Disentanglement of Pose and Expression for General Video Portrait Editing},
  author={Pang, Youxin and Zhang, Yong and Quan, Weize and Fan, Yanbo and Cun, Xiaodong and Shan, Ying and Yan, Dong-ming},
  journal={arXiv preprint arXiv:2301.06281},
  year={2023}
}
```

## Acknowledgement
Part of the code is adapted from 
[LIA](https://github.com/wyhsirius/LIA),
[PIRenderer](https://github.com/RenYurui/PIRender),
[STIT](https://github.com/rotemtzaban/STIT).
We thank authors for their contribution to the community.
