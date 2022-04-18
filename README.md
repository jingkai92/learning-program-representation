# Learning Program
Semantics with Code Representations: An Empirical Study

This repository contains the code and data in our paper, "Learning Program
Semantics with Code Representations: An Empirical Study"
published in SANER'2022. It includes POJ104Clone and POJ dataset. 

* Clone Detection - Pairwise Clone Detection
* Code Classification - Classify Code in their respective label
* Vulnerability Detection - See Devign

## Dataset

I had uploaded the dataset to google drive. You can download it [here](https://drive.google.com/file/d/1U4xQnrbym8T8QjGzTRIqqPdx7VwPP3H2/view?usp=sharing)

## Train

You can train the model with the sample command:
```shell script
python3 -u /home/jingkai/projects/cit/train.py --config_path ./ymls/clone_detection/tfidf/naivebayes.yml
```
Please look into `./ymls/<tasks>/*.yml` for setting the configurations.

## Citation
If you find this repository useful in your research, please consider citing it:
```
@inproceedings{siow2022learning,
  title={Learning Program Semantics with Code Representations: An Empirical Study},
  author={Jing Kai, Siow and Shangqing, Liu and Xiaofei, Xie and Guozhu, Meng and Yang, Liu},
  booktitle={Proceedings of the 29th IEEE International Conference onSoftware Analysis, Evolution and Reengineering},
  year={2022}
}
```