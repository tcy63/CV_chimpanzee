# CV for Primates

## Detection

### Data Preparation
First, download the images and labels for detection from [dataset for detection](https://disk.pku.edu.cn:443/link/F87EB30E935982FC8082B81D88DDE55C) and decompressed it. It is structured according to yolov5's required format.

```bash
cd datasets
tar -zvxf yolo_detection.tar.gz
```

yolov5 assumes that the directory is sturctured as follows:
```
- datasets
    - yolo_detection
        - images
            - train
                - <name>.jpg
                - ...
            - val
        - labels
            - train
                - <name>.txt
                - ...
            - val
- yolov5
```

Second, change the `datasets/chimpanzee.yaml` file according to your needs. You can change the `path` as it is the root dirctory of your dataset. It can be a path relative to the `yaml` file or a absolute path.

**NOTE**: yolov5 will substitute `images` with `labels` automatically to find the corresponding label path. Thus, for your convenience, you'd better not change the directory name.
```yaml
path: ../yolo_detection
train: images/train
val: images/val
test: # optional

names:
  0: chimpanzee
```

### Training
You can train a model from scratch:
```bash
python yolov5/train.py --data datasets/chimpanzee.yaml --weights '' --cfg yolov5s.yaml --img 640
```
Or you can use the pre-trained weights and finetune the whole network:
```bash
python yolov5/train.py --data datasets/chimpanzee.yaml --weights yolov5s.pt --img 640
```
For more details, please refer to the yolov5 GitHub page.

## Identification
### Data Preparation
First, download the dataset from [cropped images for identification](https://disk.pku.edu.cn:443/link/4DE85DF2CC9B6F655615FF26A1D9E853) and decompress it.
Here, 'crop' implies that the images used for identification are groung-truth bounding boxes.

```bash
cd datasets
tar -zvxf crop_identification.tar.gz
```
After decompression, the directory looks like this:
```
- datasets
    - yolo_detection
    - crop_identification
        - train
            - Azibo
                - 0.jpg
                - ...
            - Bambari
            - ...
            - Tai
        - val
```

### Methods
#### 1. Supervised Training with Large Margin Cosine Loss (LMCL)
##### 1) Background
***Large Margin Cosine Loss*** was first proposed by [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_CosFace_Large_Margin_CVPR_2018_paper.pdf). It aims to learn discriminative features by maximizing inter-class cosine margin. Our work first adopt this loss function in chimpanzee recognition and get relatively good results.

During training, the weight feature $W$ is trained together with the backbone model. This loss is formalized as follows:
```math
L_{lmc} = \frac{1}{N}\sum_{i=1}^N -\log \frac{e^{s(\cos(\theta_{y_i},i)-m)}}{e^{s(\cos(\theta_{y_i},i)-m)} + \sum_{j\neq y_i} e^{s(\cos(\theta_{j},i))}} 
```
```math
\cos (\theta_j, i) = W^T_j \cdot x_i, W_j = \frac{W^*_j}{||W^*_j||}, x_i = \frac{x^*_i}{||x^*_i||}
```

During testing, the class of largest cosine similarity is the predicted label.
```math
\text{Pred}_i = \text{argmax}_j (W^T_j \cdot x_i)
```
There are two hyperparameters in this loss: the forced margin $m$ and feature norm $s$.

##### 2) Model
During training and testing, the model is sturctured as follows:
```
- encoder: resnet50 -> feat_dim = 2048
- weight: W -> num_classes = 17
    - normalized Linear (2048, 17)
```
The model is trained end-to-end and validated on-the-fly.

##### 3) Training

First, change the file `identification/configs/train_lmcl.yaml` according to your needs. 

For example, `load_pt_encoder` indicates whether to use the pre-trained weights of the encoder provided by PyTorch. 

```yaml
# Part of the yaml file as an example
model: lmcl
model_args:
    encoder: resnet50
    load_pt_encoder: True  # whether to load the pre-trained weights from PyTorch
    num_classes: 17
```

Then, train the model using the configuration in the `yaml` file.
```
python identification/train.py --config identification/configs/train_lmcl.yaml
```
NOTE: you should be careful with your current working directory and the path set in the `yaml` file.

You will get the output every `print_freq` epochs like this:
```
epoch 20, train time 2.26, train_loss 11.95, train_acc 31.12; val_loss 10.80, val_acc 46.40
epoch 40, train time 2.04, train_loss 10.69, train_acc 38.37; val_loss 11.21, val_acc 43.53
epoch 60, train time 2.02, train_loss 9.71, train_acc 44.11; val_loss 10.35, val_acc 55.76
epoch 80, train time 2.23, train_loss 9.21, train_acc 49.40; val_loss 9.97, val_acc 56.47
epoch 100, train time 2.10, train_loss 9.34, train_acc 49.85; val_loss 11.35, val_acc 55.76
...
```

#### 2. Supervised Contrastive Learning
##### 1) Background
Contrastive learning is commonly used for self-supervised learning. This work proposes a new contrastive loss by leveraging label information. 
##### 2) Model
In this setup, we use a two-stage pipeline.

During training:
```
- encoder: resnet50 -> 2048
- projection head: mlp/linear -> 128
- loss: supcon
```
After training, we can use the pre-trained encoder and finetune a linear classifier on top of it.

During finetuning:
```
- (frozen) encoder: resnet50 -> 2048
- classifier: Linear -> 17
```

##### 3) Training

1. Self-supervised contrastive learning
   
    You need to specify the `model` as `supcon` and `method` as `simclr`.
    ```
    python identification/train.py --config identification/configs/train_simclr.yaml
    ```
2. Supervised contrastive learning
   
   You need to specify both the `model` and `method` as `supcon`.
    ```
    python identification/train.py --config identification/configs/train_supcon.yaml
    ```

##### 4) Finetuning
During finetuning, we will add a classifier on top of the encoder used in the training time and only train the classifier.

The most important thing is to specify the path where you stored the trained SupCon models.

For example, `identification/configs/finetune_supcon.yaml`
```yaml
# Part of the yaml file as an example
model: supcon
model_args:
    encoder: resnet50
    load_pt_encoder: False  # whether load the pre-trained weights from PyTorch
    head: mlp
    feat_dim: 128
# the path to load the trained SupCon model during the training time
load: identification/save/supcon_models/model_supcon_load_pt_encoder_True_optimizer_adam_bs_256_scheduler_exp/ckpt_epoch_300.pth
```

Then, finetune the model using this file:
```
python identification/finetune.py --config identification/configs/finetune_lmcl.yaml
```

## Pose Estimation
