# *Modality Disentangled Discriminator for Text-to-Image Synthesis*

### Introduction
This project page provides pytorch code that implements the paper: "Modality Disentangled Discriminator for Text-to-Image Synthesis".

### How to use

**Python**

- Python2.7
- Pytorch1.0+
- tensorflow (`pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.12.0-cp27-none-linux_x86_64.whl`)
- `pip install easydict pathlib`
- `conda install requests nltk pandas scikit-image pyyaml`


**Data**
1. Download metadata for [birds](https://drive.google.com/open?id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ) [coco](https://drive.google.com/open?id=1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9) and save them to `data/`
    - `python google_drive.py 1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ ./data/bird.zip`
    - `python google_drive.py 1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9 ./data/coco.zip`

2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`
    - `cd data/birds`
    - `wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz`
    - `tar -xvzf CUB_200_2011.tgz`
    
3. Download [coco](http://cocodataset.org/#download) dataset and extract the images to `data/coco/`
    - `cd data/coco`
    - `wget http://images.cocodataset.org/zips/train2014.zip`
    - `wget http://images.cocodataset.org/zips/val2014.zip`
    - `unzip train2014.zip`
    - `unzip val2014.zip`
    - `mv train2014 images`
    - `cp val2014/* images`

**Pretrained Models**
- [DAMSM for bird](https://drive.google.com/open?id=1GNUKjVeyWYBJ8hEU-yrfYQpDOkxEyP3V). Download and save it to `DAMSMencoders/`
    - `python google_drive.py 1GNUKjVeyWYBJ8hEU-yrfYQpDOkxEyP3V DAMSMencoders/bird.zip`
- [DAMSM for coco](https://drive.google.com/open?id=1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ). Download and save it to `DAMSMencoders/`
    - `python google_drive.py 1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ DAMSMencoders/coco.zip`
- [DM-GAN-MDD for bird](https://drive.google.com/file/d/1TnKf-SKG06VxkvmpUe2anEIbOD5QsAva/view?usp=sharing). Download and save it to `models`
- [DM-GAN-MDD for coco](https://drive.google.com/file/d/1eLHo-lM5-npx9RANdgKLmK_LiaoRBFsD/view?usp=sharing). Download and save it to `models`
- [IS for bird](https://drive.google.com/file/d/0B3y_msrWZaXLMzNMNWhWdW0zVWs)
    - `python google_drive.py 0B3y_msrWZaXLMzNMNWhWdW0zVWs eval/IS/inception_finetuned_models.zip`
- [FID for bird](https://drive.google.com/file/d/1747il5vnY2zNkmQ1x_8hySx537ZAJEtj)
    - `python google_drive.py 1747il5vnY2zNkmQ1x_8hySx537ZAJEtj eval/FID/bird_val.npz`
- [FID for coco](https://drive.google.com/file/d/10NYi4XU3_bLjPEAg5KQal-l8A_d8lnL5)
    - `python google_drive.py 10NYi4XU3_bLjPEAg5KQal-l8A_d8lnL5 eval/FID/coco_val.npz`

**Training**
- go into `code/` folder
- bird: `python main.py --cfg cfg/bird_DMGANMDD.yml --gpu 0`
- coco: `python main.py --cfg cfg/coco_DMGANMDD.yml --gpu 0`

**Validation**
1. Images generation:
    - go into `code/` folder  
    - `python main.py --cfg cfg/eval_bird_DMGANMDD.yml --gpu 0`
    - `python main.py --cfg cfg/eval_coco_DMGANMDD.yml --gpu 0`
2. Inception score ([IS for bird](https://github.com/hanzhanggit/StackGAN-inception-model), [IS for coco](https://github.com/openai/improved-gan/tree/master/inception_score)):
    - `cd DM-GAN-MDD/eval/IS/bird && CUDA_VISIBLE_DEVICES=0 python inception_score_bird.py --image_folder ../../../models/netG_DMGANMDD_bird`
    - `cd DM-GAN-MDD/eval/IS/coco && CUDA_VISIBLE_DEVICES=0 python inception_score_coco.py ../../../models/netG_DMGANMDD_coco`
3. FID:
    - `cd DM-GAN-MDD/eval/FID && python fid_score.py --gpu 0 --path1 bird_val.npz --path2 ../../models/netG_DMGANMDD_bird`
    - `cd DM-GAN-MDD/eval/FID && python fid_score.py --gpu 0 --path1 coco_val.npz --path2 ../../models/netG_DMGANMDD_coco`

**Performance**

As DM-GAN, we use the Pytorch implementation to measure FID score. 

|Model |R-precision↑  |IS↑  |[Pytorch FID](https://github.com/mseitzer/pytorch-fid/tree/802da3963113b5b5f8154e0e27580ee4c97460ab)↓ |
|----|-----| -----|---|
| bird_AttnGAN (paper) | 67.82% ± 4.43%| 4.36 ± 0.03| 23.98|
| bird_DMGAN (paper) | 72.31% ± 0.91%| 4.75 ± 0.07| 16.09|
| bird_DMGAN_MDD | 79.73% ± 0.68%| 4.86 ± 0.06| 15.76|
| coco_AttnGAN (paper) | 85.47% ± 3.69%| 25.89 ± 0.47 | 35.49|
| coco_DMGAN (paper) | 88.56% ± 0.28%| 30.49 ± 0.57 | 32.64|
| coco_DMGAN_MDD | 94.37% ± 0.36%| 34.46 ± 0.72 | 24.30|

### License
This code is released under the MIT License (refer to the LICENSE file for details). 
