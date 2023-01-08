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

### Training
Take the *Large Margin Cosine Loss* method as an example.

First, change the file `contrast/configs/train_lmcl.yaml` according to your needs.
For example, you can choose whether to use the pre-trained encoder from PyTorch.

Then, train the model using the configuration in the `yaml` file.
```
python contrast/train.py --config contrast/configs/train_lmcl.yaml
```
NOTE: you should be careful with your current working directory and the path set in the `yaml` file.
## Pose Estimation
