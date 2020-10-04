# How to run run.sh file
# We give multiple options!

# Examples....
# To run Standard training 
# CUDA_VISIBLE_DEVICES=? ./run.sh Standard

# To run adversarial training
# CUDA_VISIBLE_DEVICES=? ./run.sh adv_train_linf

# To run test
# CUDA_VISIBLE_DEVICES=? ./run.sh test ../checkpoint/ckpt.t7....
echo "CUDA ./total_process.sh test ../checkpoint/ name model learning rate"

if [ "$1" == "test" ]; then
	
	python -m torch.distributed.launch --nproc_per_node=1 linear_eval.py --ngpu 1 --batch-size=1024 --train_type='linear_eval' --model=$4 --epoch 150 --lr $5 --name $3 --load_checkpoint=$2 --clean=True --dataset=$6 --seed=$7
	
	python -m torch.distributed.launch --nproc_per_node=1 robustness_test.py --ngpu 1 --train_type='linear_eval' --name=$3 --batch-size=1024 --model=$4 --load_checkpoint='../checkpoint/ckpt.t7'$3'_Evaluate_linear_eval_ResNet18_'$6'_'$7 --attack_type='linf' --epsilon=0.0314 --alpha=0.00314 --k=20 --module --dataset=$6 --seed=$7
        python -m torch.distributed.launch --nproc_per_node=1 robustness_test.py --ngpu 1 --train_type='linear_eval' --name=$3 --batch-size=1024 --model=$4 --load_checkpoint='../checkpoint/ckpt.t7'$3'_Evaluate_linear_eval_ResNet18_'$6'_'$7 --attack_type='linf' --epsilon=0.0627 --alpha=0.00627 --k=20 --module --dataset=$6 --seed=$7

elif [ "$1" == "advtest" ]; then
	python -m torch.distributed.launch --nproc_per_node=1 linear_eval.py --ngpu 1 --batch-size=1024 --train_type='linear_eval' --model=$4 --epoch 150 --lr $5 --name $3 --load_checkpoint=$2 --adv_img=True
	python -m torch.distributed.launch --nproc_per_node=1 robustness_test.py --ngpu 1 --train_type='linear_eval' --name=$3 --batch-size=1024 --model=$4 --load_checkpoint='../checkpoint/ckpt.t7'$3'_Evaluate_linear_eval_ResNet18_'$6'_'$7 --attack_type='linf' --epsilon=0.0314 --alpha=0.00314 --k=20 --module --dataset=$6 --seed=$7
        python -m torch.distributed.launch --nproc_per_node=1 robustness_test.py --ngpu 1 --train_type='linear_eval' --name=$3 --batch-size=1024 --model=$4 --load_checkpoint='../checkpoint/ckpt.t7'$3'_Evaluate_linear_eval_ResNet18_'$6'_'$7 --attack_type='linf' --epsilon=0.0627 --alpha=0.00627 --k=20 --module --dataset=$6 --seed=$7

elif [ "$1" == "linear" ]; then
	python -m torch.distributed.launch --nproc_per_node=1 linear_eval.py --ngpu 1 --batch-size=1024 --train_type='linear_eval' --model=$4 --epoch 150 --lr $5 --name $3 --load_checkpoint=$2
	
elif [ "$1" == "adv_check" ]; then
	python -m torch.distributed.launch --nproc_per_node=1 robustness_test.py --ngpu 1 --train_type='linear_eval' --name=$3 --batch-size=1024 --model=$4 --load_checkpoint='../checkpoint/ckpt.t7'$3'_Evaluate_linear_eval_ResNet18_'$6'_'$7 --attack_type='linf' --epsilon=0.0314 --alpha=0.00314 --k=20 --module --dataset=$6 --seed=$7
        python -m torch.distributed.launch --nproc_per_node=1 robustness_test.py --ngpu 1 --train_type='linear_eval' --name=$3 --batch-size=1024 --model=$4 --load_checkpoint='../checkpoint/ckpt.t7'$3'_Evaluate_linear_eval_ResNet18_'$6'_'$7 --attack_type='linf' --epsilon=0.0627 --alpha=0.00627 --k=20 --module --dataset=$6 --seed=$7

else
    echo -e "\nValueError: Unsupported mode!!!"
    echo "There are modes: Standard, adv_train_linf, TRADES_linf, LLR_linf, test, saliency, loss"  
    echo -e "The first argument should be one of these modes.\n"
fi
