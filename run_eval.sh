export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4
DATA_PATH=../datasets/MSRVTT
RPort=$(shuf -i 1000-9999 -n1)
Margin=0.1
beta=0.2
CKPT_NAME=ckpt_msrvtt_woFreeze_0321
Tau=1.0
epoch=$(seq 3 3)
for ep in $epoch
do
    OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=9 python -m torch.distributed.launch --master_port $RPort --nproc_per_node=1 main_task_retrieval.py --do_eval --num_thread_reader=48 \
        --epochs=5 --batch_size=32 --n_display=50 --train_csv ${DATA_PATH}/MSRVTT_train.9k.csv --val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
        --data_path ${DATA_PATH}/MSRVTT_data.json --features_path ${DATA_PATH}/videos/all_compressed --audio_path ${DATA_PATH}/videos/audio_all_compressed --output_dir ckpts/${CKPT_NAME} --lr 1e-4 \
        --max_words 32 --max_frames 12 --batch_size_val 100  --datatype msrvtt --expand_msrvtt_sentences  --feature_framerate 1 --coef_lr 1e-3 --freeze_layer_num 12  \
        --slice_framepos 2 --loose_type --linear_patch 2d --sim_header seqTransf --pretrained_clip_name ViT-B/32 --eval_max_frame 12 --temperature $Tau --cross_num_hidden_layers 4 --audio_query_layers 4 --beta $beta --margin_BD $Margin --init_model ckpts/${CKPT_NAME}/pytorch_model.bin.$ep
done
chmod -R 777 ckpts/*


