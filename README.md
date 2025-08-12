# AVIGATE: Learning Audio-Guided Video Representation with Gated Attention for Video-Text Retrieval (CVPR 2025, Oral)


This repository is an official implementation of the paper [**AVIGATE: Learning Audio-Guided Video Representation with Gated Attention for Video-Text Retrieval**](https://openaccess.thecvf.com/content/CVPR2025/papers/Jeong_Learning_Audio-guided_Video_Representation_with_Gated_Attention_for_Video-Text_Retrieval_CVPR_2025_paper.pdf). 
![AVIGATE](AVIGATE.png)

## Requirement
```sh
# From CLIP
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install opencv-python boto3 requests pandas
```
### Conda Environment
```sh
conda env create --file video.yml
```
## Data Preparing

**For MSRVTT**

The official data and video links can be found in [link](http://ms-multimedia-challenge.com/2017/dataset). 

For the convenience, you can also download the splits and captions by,
```sh
wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip
```

Besides, the raw videos can be found in [sharing](https://github.com/m-bain/frozen-in-time#-finetuning-benchmarks-msr-vtt) from *Frozen️ in Time*, i.e.,
```sh
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
```
For videos without audio signals, we obtained audio sources using external crawling tools like [youtube-dl](https://github.com/yt-dlp/yt-dlp).  
We get 9,582 audio signals for 10,000 videos.

## Compress Video for Speed-up (optional)
```sh
python preprocess/compress_video.py --input_root [raw_video_path] --output_root [compressed_video_path]
```
This script will compress the video to *3fps* with width *224* (or height *224*). Modify the variables for your customization.

# How to Run
Download CLIP (ViT-B/32) weight,
```sh
wget -P ./modules https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```
or, download CLIP (ViT-B/16) weight,
```sh
wget -P ./modules https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```
Download AST weight from [AST](https://github.com/YuanGongND/ast) (Pretrained Models 1: "Full AudioSet, 10 tstride, 10 fstride, with Weight Averaging (0.459 mAP)").


**For MSR-VTT Training** 
```sh
run.sh
```
**For MSR-VTT Evaluation** 
```sh
run_eval.sh
```
# Citation
If you find CLIP4Clip useful in your work, you can cite the following paper:
```bibtex
@InProceedings{Jeong_2025_CVPR,
    author    = {Jeong, Boseung and Park, Jicheol and Kim, Sungyeon and Kwak, Suha},
    title     = {Learning Audio-guided Video Representation with Gated Attention for Video-Text Retrieval},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {26202-26211}
}
```

# Acknowledgments
Our code is based on [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip) and [AST](https://github.com/YuanGongND/ast).
