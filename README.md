## Joint Learning of Intrinsic Images and Semantic Segmentation

We provide the inference code for our paper titled Joint Learning of Intrinsic Images and Semantic Segmentation, ECCV 2018.

The model is based on the architecture of 'Learning Non-Lambertian Object Intrinsics across ShapeNet Categories' by Shi et. CVPR17.

The scripts have been tested on PyTorch 1.0.

## Semantic Labels
```
0.  void
1.  ground
2.  grass
3.  dirt
4.  gravel
5.  mulch
6.  pebble
7.  woodchip
8.  pavement
11. box
20. topiary
30. rose
102. tree
103. fence
104. step
105. flowerpot
106. stone
223. sky
```

## Requirements
The code has been written for Python 3. The requirements are as follows:

    * Pytorch
    * Pillow
    * TQDM (really versatile progressbar library)
    
    
We highly recommend using the Anaconda distribution for easy installing all the required libraries. Creating a separte environment for this is also recommended.
You can install the requirements using the requirements.txt file. Due to how conda packages are distributed, some of the packages need a specific channel to be specified, in order to have the latest version of the package. Unfortunately, there is no way of doing it using the requirements.txt, so they need to be manually fed in like follows:

```bash
conda install tqdm pytorch torchvision cudatoolkit=10.0 -c pytorch
```

## Inference

The script takes the following arguments:\
&nbsp;&nbsp;&nbsp;&nbsp;--_file_ : A folder or the target file for the decomposition. If it is a folder, then the script will look for png files in the folder. However it is not recursive and will only look for the files on the top level. The number of found files will also be reported. **This is a required argument.**\
&nbsp;&nbsp;&nbsp;&nbsp;--_name_ : Name of the experiment. This is synthetic\_trained for the provided model (as shown in the above structure). As a rule of thumb, this is folder name above the checkpoints folder. **This is a required argument.**\
&nbsp;&nbsp;&nbsp;&nbsp;--_model_ : This is the model store location. This is the folder above the experiment folder in the model provided, so the _current directory_ in the structure below. The script will search for the _experiment_ folder in this location. **This is a required argument.**\
&nbsp;&nbsp;&nbsp;&nbsp;--_gpu_ : This is the ID of the GPU on your machine. Default value is 0, which is the first GPU in the system. The model requires about 1GB of video memory for the inference.\
&nbsp;&nbsp;&nbsp;&nbsp;--_results_ : The folder name for the result storage location. Default is _results_ folder in the same directory as the _infer.py_ file.


Here, the expected folder structure is as follows:\

```bash
<Download Location>
├── direct\_intrinsics\_sn.py
├── experiment
│   └── synthetic\_trained
│       ├── checkpoints
│       │   └── final.checkpoint
│       └── model.weights
├── infer.py
├── README.md
├── requirements.txt
└── utils.py
```

Run the script as follows:
```python
    python infer.py --name synthetic_trained --file test.png --gpu 0 --model ./
```

# Real World Garden Semantics
As done in the paper, the model can be fine-tuned to generalize the semantic segmentation results on the real world gardens by using the dataset provided [here](http://trimbot2020.webhosting.rug.nl/events/3drms/challenge/).

For further modifications, the script has been commented, along with verbose print statements. So modifications should be trivial.

## Citation

If you use this work, please cite our paper like follows:

```
@inproceedings{Baslamisli2018ECCV,
 author = {A. ~S. Baslamisli and T. ~T. Groenestege and P. Das and H. ~A. Le and S. Karaoglu and T. Gevers},
 title = {Joint Learning of Intrinsic Images and Semantic Segmentation},
 booktitle = {European Conference on Computer Vision},
 year = 2018
}
```
