args=(
    # Dataset options
    --dataset './datasets/lausanne' --timestep 15 --obs_len 8 --loader_num_workers 1
    # Optimization
    --num_epochs 200 --batch_size 32
    # Model options
    --embedding_dim 128 --h_dim 32 --num_layers 1 --mlp_dim 64 --dropout 0.5 --batch_norm 0 --learning_rate 0.0005
    # Output
    --output_dir './models' --print_every 100 --checkpoint_every 10 --checkpoint_name 'cnnlstm_lausanne_standardcrop' --checkpoint_start_from None --restore_from_checkpoint 1
    # Misc
    --use_gpu 1 --gpu_num '0'
    )

python3 train.py "${args[@]}"

