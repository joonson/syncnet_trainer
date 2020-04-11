## SyncNet trainer

This repository contains the baseline code for the VoxSRC 2020 self-supervised speaker verification track.

#### Dependencies
```
pip install -r requirements.txt
```

#### Data preparation

The [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) datasets are used for these experiments.

Once downloaded, the following script can be used to generate the training and test lists.

```
python ./makeFileList.py --output data/dev.txt --mp4_dir VOX2_PATH/dev/mp4 --txt_dir VOX2_PATH/dev/txt --wav_dir VOX2_PATH/dev/wav
python ./makeFileList.py --output data/test.txt --mp4_dir VOX2_PATH/test/mp4 --txt_dir VOX2_PATH/test/txt --wav_dir VOX2_PATH/test/wav
```

Check that the training list contains approximately 1,092K lines and the test list 36K lines.

#### Training example

```
python trainSyncNet.py --temporal_stride 2 --maxFrames 50 --model models.SyncNetModelFBank --save_path data/exp01 
```

#### Pretrained model

A pretrained model can be downloaded from [here](http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/voxsrc2020_baseline.model).

You can check that the following script returns: `EER 20.4348`.

```
python trainSyncNet.py --temporal_stride 2 --maxFrames 50 --model models.SyncNetModelFBank --initial_model data/voxsrc2020_baseline.model --eval 
```

#### Citation

The models are trained with the `Identity loss + Content loss` described in the paper below. The only difference here is that the scaled cosine distance is used instead of the Euclidean distance.

Please cite the following if you make use of the code.

```
@InProceedings{Nagrani20d,
  author       = "Arsha Nagrani and Joon~Son Chung and Samuel Albanie and Andrew Zisserman",
  title        = "Disentangled Speech Embeddings using Cross-Modal Self-Supervision",
  booktitle    = "International Conference on Acoustics, Speech, and Signal Processing",
  year         = "2020",
}
```

#### License
```
Copyright (c) 2020-present Visual Geometry Group.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
