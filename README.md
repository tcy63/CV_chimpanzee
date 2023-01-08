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
For more details, please refer to the yolov5 GitHub page [...]

## Identification

## Pose Estimation
