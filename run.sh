set -ex

cuda=1
ex_name=2to1_vgg19
crop_size=256
load_size=286
ATTACK=20 


target=1  # 30 speed limit

input_label=2 # 50 speed limit
k_spt=1
k_qry=1
finetune_step=30
dist=30
update_step=10
save_latest_freq=1000
task_num=1

python train.py --meta_dataroot "image" --update_step $update_step --gpu_ids $cuda --lambda_ATTACK_B $ATTACK  \
--k_spt $k_spt --k_qry $k_qry --ori $input_label --target $target --name $ex_name --crop_size $crop_size  \
--load_size $load_size --finetune_step $finetune_step --lambda_dist $dist --task_num $task_num \
--save_latest_freq $save_latest_freq 
