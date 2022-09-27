# ProbKT

ProbKT, a framework based on probabilistic reasoning to train object detection models with weak supervision, by transferring knowledge from a source domain where rich image annotations are available.

## Prerequisites and installation

To take advantage of full features of code we recommend to create an account on [WANDB](https://wandb.ai/) and login.

ProbKT finetuning also depends on [DeepProbLog](https://github.com/ML-KULeuven/deepproblog).

For easy of use we recommend to also first install and set up a [poetry environment](https://python-poetry.org)

Then execute:

``
poetry install
``


## Get Datasets
All datasets can be downloaded with instructions [Here](datasets/README.md)

For setting up for the MNIST experiments you should execute:

```
cd generate_data
wget --no-check-certificate -O mnist.tar.gz https://figshare.com/ndownloader/files/35142142?private_link=c760de026f000524db5a
tar -xvzf mnist.tar.gz
```

## Train Baseline model

For training the baseline model on the MNIST dataset execute:

```
poetry run python robust_detection/baselines/train.py --data_dir mnist/mnist3_all
```

## Pretrain RCNN Model

Pretrain the RCNN model on source domain of MNIST dataset:

```
poetry run python robust_detection/train/train_rcnn.py --data_path mnist/mnist3_skip
```

## Pretrain DETR Model

Pretrain the DETR model on source domain of MNIST dataset:

```
poetry run python robust_detection/train/train_detr.py --data_path mnist/mnist3_skip --rgb True
```

## ProbKT Finetune RCNN Pretrained model

For finetuning a sweep is assumed on your wandb account for the 5 fold pretrained RCNN model. Example sweep configuration:
```
method: grid
parameters:
  batch_size:
    values:
      - 1
  data_path:
    values:
      - mnist/mnist3_skip
  epochs:
    values:
      - 30
  fold:
    values:
      - 0
      - 1
      - 2
      - 3
      - 4
  pre_trained:
    values:
      - true
program: train_rcnn.py
```

Once sweep has ran succesful finetuning can start using:

```
poetry run python robust_detection/train/train_fine_tune.py --og_data_path mnist/mnist3_skip --target_data_path mnist/mnist3_all --agg_case True --fold 0 --sweep_id <sweepid>
```

# WSOD_transfer 

For the experiments with the CLEVR-mini and Molecules data sets it is needed to use this [fork](https://github.com/iclr2023-ProbKT/wsod_transfer).

## Instructions for applying WSOD_transfer on CLEVR-mini data set(1-fold)

```
python  tools/train_net.py --config-file wsod/clevr_skip_to_all/ocud_it0.yaml
python  tools/train_net.py --config-file wsod/clevr_skip_to_all/mil_it0.yaml
python wsod/pseudo_label_objects.py output/clevr_skip_to_all_long_fold_0/mil_it0/inference/clevr_all_nofold clevr 0 0.8
python  tools/train_net.py --config-file wsod/clevr_skip_to_all/ocud_it1.yaml --start_iter 0
python  tools/train_net.py --config-file wsod/clevr_skip_to_all/mil_it1.yaml --start_iter 0
python wsod/pseudo_label_objects.py output/clevr_skip_to_all_long_fold_0/mil_it1/inference/clevr_all_nofold clevr 0 0.8
python  tools/train_net.py --config-file wsod/clevr_skip_to_all/ocud_it2.yaml --start_iter 0
python  tools/train_net.py --config-file wsod/clevr_skip_to_all/mil_it2.yaml --start_iter 0
python wsod/pseudo_label_objects.py output/clevr_skip_to_all_long_fold_0/mil_it2/inference/clevr_all_nofold clevr 0 0.8
python  tools/train_net.py --config-file wsod/clevr_skip_to_all/distill_resnet50c4.yaml
```

## Instructions to evaluate performance

```
python tools/create_np_preds.py --preds_file output/clevr_skip_to_all_long_fold_0/mil_it2/distill_resnet50/inference/clevr_all_test/predictions.pth --outputfile preds_clevr_all_long_fold_0.npy
```
then in the ProbKT environment:
```
python tools/evaluate_preds.py --preds_file preds_clevr_skip_long_fold_0.npy --true_data_dir ../ProbKT/generate_data/clevr/clevr_skip_cube/test/
```
