# for k in {1..2001..50}
#   do
#     echo "$k"
#     # ./TEST/main_sequential $k  517
#     # ./cuda/main_cuda_v1 $k  517
#     # ./cuda/main_cuda_v2 $k  517
#     # ./cuda/main_cuda_v3 $k  517
#   done

for n in {100..20000..100}
  do
    echo "$n"
    # ./TEST/main_sequential 11 $n
    # ./cuda/main_cuda_v1 11 $n
    # ./cuda/main_cuda_v2 11 $n
    ./cuda/main_cuda_v3 4 $n
  done
