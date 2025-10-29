# GNEprop with C++ Acceleration
This work implements GNEprop with C++ acceleration.

## Set up Environment
You can either setup your environment using docker or conda.

### 1. Setup using Docker

Build your docker container use the file [Dockerfile.conda](docker/Dockerfile.conda)
```
docker build -t <img_name> -f Dockerfile.conda .
```

### 2. Setup using conda
The python environment is managed by `conda`. 
1. Install Conda on your system if you haven't already.

2. Set up the environment using [setup_env.sh](setup_env.sh) by running:
```
sh setup_env.sh
```

## Commands
### Usage
To enable C++ acceleration with the GNEpropCPP module, use the ```code_version 2``` flag in your commands.

### Pretraining
Run the pretraining process with:
```
python3 clr.py --dataset_path <your_dataset_path> --gpus 1 --num_workers 3 --max_epoch 2 --batch_size 4096 --lr 1e-03 --model_hidden_size 500 --model_depth 5 --weight_decay 0. --exclude_bn_bias --project_output_dim 256 --code_version 2
```

### Inference
For inference, use:
```
python3 scripts/virtual_screening.py --model_path <your_checkpoint_dir> --output_dir <your_output_dir> --data_file_path <your_dataset_path> --gpus 1 --num_workers 3 --batch_size 256 --code_version 2
```

## Example
Please reference the [Makefile](docker/Makefile) for a test example in Docker environment.






