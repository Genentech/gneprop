COMMIT := $(shell git log -n 1 --pretty=format:"%H" | head -c 8)
IMG := gneprop-cpp:$(COMMIT)
NGC := gneprop-cpp-ngc:$(COMMIT)
CURRENT_USER := -v /tmp/$(USER):/home/$(USER) -e HOME=/home/$(USER) -v /etc/passwd:/etc/passwd -u $(shell id -u):$(shell id -g)
LOGS := -v /tmp/gneprop_logs:/tmp/gneprop_logs
SAMPLE_DATA := -v /tmp/support_data:/workspace/support_data
CUR_DIR := $(shell dirname "$(realpath $(firstword $(MAKEFILE_LIST)))")
ifdef NV_GPU
	GPUS := --gpus \"device=$${NV_GPU}\"
else
	GPUS := --gpus 1
endif

container:
	docker build -t $(IMG) -f Dockerfile.conda ..
	touch .container

container_ngc:
	docker build -t $(NGC) -f Dockerfile.ngc ..
	touch .container

support_data: | .container /tmp/$(USER)
	[ -e /tmp/support_data ] && rm -rf /tmp/support_data || true
	mkdir /tmp/support_data
	docker run --rm -it $(SAMPLE_DATA) $(CURRENT_USER) $(IMG) bash -lc "gdown --no-cookies https://drive.google.com/drive/folders/1xyXa90VSEubpeRg0c6TcgZRTOYpPtSXm -O support_data --folder"
	ln -s /tmp/support_data .

run:
	docker run --rm -it $(CURRENT_USER) $(GPUS) $(IMG) bash -l
run_ngc:
	docker run --rm -it $(CURRENT_USER) $(GPUS) $(NGC) bash -l

/tmp/gneprop_logs:
	mkdir /tmp/gneprop_logs
/tmp/$(USER):
	mkdir $@

# A similar command line was run with --gpus 8 --num_workers 16 on a system with 8 A100 GPUs.
# --code_version 0 is the original Python code before optimizing
# --code_version 2 uses the C++ code in GNEpropCPP/GNEpropCPP.cpp
test_pretraining: | support_data /tmp/gneprop_logs /tmp/$(USER)
	docker run --runtime nvidia --privileged --rm -it --user $(id -u):$(id -g) --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 $(SAMPLE_DATA) $(CURRENT_USER) $(GPUS) $(LOGS) -v $(CUR_DIR)/..:/local $(IMG) bash -lc "python3 clr.py \
		--dataset_path <dataset_path> \
		--gpus 1 --num_workers 3 --max_epoch 2 --batch_size 4096 \
		--lr 1e-03 --model_hidden_size 500 --model_depth 5 \
		--weight_decay 0. --exclude_bn_bias \
		--project_output_dim 256 --code_version 2"
# A similar command line was run with --gpus 8 --num_workers 16 on a system with 8 A100 GPUs.
# --code_version 0 is the original Python code before optimizing
# --code_version 2 uses the C++ code in GNEpropCPP/GNEpropCPP.cpp
test_inference: | support_data /tmp/gneprop_logs /tmp/$(USER)
	docker run --runtime nvidia --privileged --rm -it --user $(id -u):$(id -g) --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 $(SAMPLE_DATA) $(CURRENT_USER) $(GPUS) $(LOGS) -v $(CUR_DIR)/..:/local $(IMG) bash -lc "python3 /local/scripts/virtual_screening.py \
		--model_path <checkpoint_dir> --output_dir <output_dir> \
		--data_file_path <data_file> \
		--gpus 1 --num_workers 3 --batch_size 4096 --code_version 2"
test_ngc: | support_data /tmp/gneprop_logs /tmp/$(USER)
	docker run --rm -it $(SAMPLE_DATA) $(CURRENT_USER) $(GPUS) $(LOGS) $(NGC) bash -lc "python gneprop_pyg.py \
		--dataset_path <dataset_path> --lr 4.9379e-05 \
		--hidden_size 500 --depth 5 --num_readout_layers 1 \
		--dropout 0.13 --lr_strategy warmup_cosine_step --aggr mean \
		--gpus 1 --split_type scaffold --max_epochs 7 \
		--metric val_ap --num_workers 3 --log_directory /tmp/gneprop_logs \
		--parallel_folds 2 --adv flag --adv_m 5"

clean:
	rm -rf /tmp/support_data support_data /tmp/gneprop_logs /tmp/$(USER)
