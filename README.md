# CR-UNIT

## Unsupervised Image-to-Image Translation with Content Reconstruction

Abstract: The goal of unsupervised image-to-image translation is to translate images between two different domains without any paired data. Recent research has shown great progress in this area, but there are still challenges when translating between artificial and natural objects. To overcome this problem, we propose a novel unsupervised image-to-image translation with content reconstruction (CR-UNIT), a staged and from-coarse-to-fine training framework. To be specific, CR-UNIT builds a corresponding relationship among the content features of different domains on a coarse granularity in the first stage. In the second stage, a new content reconstruction module is constructed to extract the fine-grained content and style features, obtaining more detailed semantic correspondence and better fusion of the content and style features. Furthermore, we design a content reconstruction loss to facilitate the training of our model. Extensive experimental results demonstrate the superiority of the proposed CR-UNIT over the existing methods. Especially for the translation task between the artificial and natural objects, our CR-UNIT achieves outstanding effect in terms of perceptive quality and objective metric.

## Installation
```bash
git clone https://github.com/gzhu-DVL/CR-UNIT.git
cd CR-UNIT
```
**Dependencies:**

We have tested on:
- CUDA 10.1
- PyTorch 1.7.0

All dependencies for defining the environment are provided in `environment/gpunit_env.yaml`.
```bash
conda env create -f ./environment/gpunit_env.yaml
```

## Dataset Preparation
| Task | Used Dataset | 
| :--- | :--- | 
| Male←→Female | [CelebA-HQ](https://github.com/clovaai/stargan-v2#datasets-and-pre-trained-networks): divided into male and female subsets by [StarGANv2](https://github.com/clovaai/stargan-v2#datasets-and-pre-trained-networks) |
| Dog←→Cat| [AFHQ](https://github.com/clovaai/stargan-v2#datasets-and-pre-trained-networks) provided by [StarGANv2](https://github.com/clovaai/stargan-v2#datasets-and-pre-trained-networks) |
| Face←→Cat| [CelebA-HQ](https://github.com/switchablenorms/CelebAMask-HQ) and [AFHQ](https://github.com/clovaai/stargan-v2#datasets-and-pre-trained-networks) |
| Bird←→Car | 4 classes of birds and 4 classes of cars in ImageNet291. Please refer to [dataset preparation](./data_preparation#2-imagenet291) for building ImageNet291 from [ImageNet](https://image-net.org/download.php) |


## Image-to-Image Translation
Translate a content image to the target domain in the style of a style image by additionally specifying `--style`:
```python
python inference.py --generator_path PRETRAINED_GENERATOR_PATH --content_encoder_path PRETRAINED_ENCODER_PATH \ 
                    --content CONTENT_IMAGE_PATH --style STYLE_IMAGE_PATH --device DEVICE
```

## Train Image-to-Image Transaltion Network
```python
python train.py --task TASK --batch BATCH_SIZE --iter ITERATIONS \
                --source_paths SPATH1 SPATH2 ... SPATHS --source_num SNUM1 SNUM2 ... SNUMS \
                --target_paths TPATH1 TPATH2 ... TPATHT --target_num TNUM1 TNUM2 ... TNUMT
```
### examples
> python train.py --task cat2dog --source_paths ./data/afhq/images512x512/train/cat --source_num 4000 --target_paths ./data/afhq/images512x512/train/dog --target_num 4000 --mitigate_style_bias

                                                            
                              
## Acknowledgments

The code is developed based on [GP-UNIT]:https://github.com/williamyang1991/GP-UNIT
