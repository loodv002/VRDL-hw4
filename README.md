# NYCU VRDL spring 2025 HW4

Student ID: 110550138

Name: 鄭博文

## Introduction

A `PromptIR` model for image restoration task, with training and inference code.

## How to Run

### Install requirements:

  ```bash
  pip install -r requirements.txt
  ```

  Note that the package versions listed in the `requirements.txt` file have been tested only on Python 3.10. For other Python versions, consider removing the specified versions and manually resolving dependencies.

### Configuration:

  Check the comments in `config-example.yml`.

### Split training and validation data:

  ```bash
  python split_train_val.py [--config <config_file_path>]
  ```
  The default value of `config` argument is `./config.yml`. The `data/train` directory will be splitted into `data/train_splitted` and `data/val_splitted` directories. 

### Training:

  ```bash
  python train.py [--config <config_file_path>]
  ```

### Inference:

  ```
  python inference.py --checkpoint <checkpoint_name> [--config <config_file_path>]
  ```
  The checkpoint name should be the stem of `.pth` file in `MODEL_DIR`, with format `{date}-{time}_epoch_{epoch}`. For example, `20250414-222522_epoch_4` (without .pth).

  For ensemble inference, run:

  ```
  python ensemble_inference.py --checkpoint <checkpoint_name> [--config <config_file_path>]
  ```

## Performance:

  PSNR: 30.03

  ![image](performance_screenshot.jpg)