# pytorch-liteflownet
This is a personal reimplementation of LiteFlowNet [1] using PyTorch. Should you be making use of this work, please cite the paper accordingly. Also, make sure to adhere to the <a href="https://github.com/twhui/LiteFlowNet#license-and-citation">licensing terms</a> of the authors. Should you be making use of this particular implementation, please acknowledge it appropriately [2].

<a href="https://arxiv.org/abs/1805.07036" rel="Paper"><img src="http://www.arxiv-sanity.com/static/thumbs/1805.07036v1.pdf.jpg" alt="Paper" width="100%"></a>

For the original Caffe version of this work, please see: https://github.com/twhui/LiteFlowNet
<br />
Another optical flow implementation from me: https://github.com/sniklaus/pytorch-pwc
<br />
And another optical flow implementation from me: https://github.com/sniklaus/pytorch-spynet
<br />
Yet another optical flow implementation from me: https://github.com/sniklaus/pytorch-unflow

## setup
To download the pre-trained models, run `bash download.bash`. These originate from the original authors, I just converted them to PyTorch.

The correlation layer is implemented in CUDA using CuPy, which is why CuPy is a required dependency. It can be installed using `pip install cupy` or alternatively using one of the provided binary packages as outlined in the CuPy repository.

## usage
To run it on your own pair of images, use the following command. You can choose between three models, please make sure to see their paper / the code for more details.

```
python run.py --model default --first ./images/first.png --second ./images/second.png --out ./out.flo
```

I am afraid that I cannot guarantee that this reimplementation is correct. However, it produced results pretty much identical to the implementation of the original authors in the examples that I tried. There are some numerical deviations that stem from differences in the `DownsampleLayer` of Caffe and the `torch.nn.functional.interpolate` function of PyTorch. Please feel free to contribute to this repository by submitting issues and pull requests.

## comparison
<p align="center"><img src="comparison/comparison.gif?raw=true" alt="Comparison"></p>

## license
As stated in the <a href="https://github.com/twhui/LiteFlowNet#license-and-citation">licensing terms</a> of the authors of the paper, their material is provided for research purposes only. Please make sure to further consult their licensing terms.

## references
```
[1]  @inproceedings{Hui_CVPR_2018,
         author = {Tak-Wai Hui and Xiaoou Tang and Chen Change Loy},
         title = {{LiteFlowNet}: A Lightweight Convolutional Neural Network for Optical Flow Estimation},
         booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
         year = {2018}
     }
```

```
[2]  @misc{pytorch-liteflownet,
         author = {Simon Niklaus},
         title = {A Reimplementation of {LiteFlowNet} Using {PyTorch}},
         year = {2019},
         howpublished = {\url{https://github.com/sniklaus/pytorch-liteflownet}}
    }
```