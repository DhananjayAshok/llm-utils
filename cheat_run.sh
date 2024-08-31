ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=2 cheat_script.py --config qlora.yaml --dataset_path data/l/ 
#python cheat_script.py --config qlora.yaml