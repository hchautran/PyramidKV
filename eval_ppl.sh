# for window_size in 1024 2048 4096
# do
#    for algo in pitome stream 
#    do
#       CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --overwrite --experiment $algo --window_size $window_size 
#    done
#    CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --overwrite --experiment baseline --window_size $window_size 
# done 
CUDA_VISIBLE_DEVICES=3 python eval_ppl.py --overwrite --experiment pitome --window_size 256 