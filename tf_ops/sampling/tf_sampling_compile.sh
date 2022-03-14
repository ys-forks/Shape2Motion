CUDA_PATH="/local-scratch/localhome/yma50/Development/spack/opt/spack/linux-ubuntu20.04-skylake_avx512/gcc-7.3.0/cuda-10.0.130-qtv6jwtpxgqcacupdfffbqq4faipoky7"
TENSORFLOW_DIR="/local-scratch/localhome/yma50/anaconda3/envs/tf1/lib/python3.7/site-packages/tensorflow_core"
#/bin/bash
$CUDA_PATH/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

echo -L$TENSORFLOW_DIR

# TF1.2
#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I $TENSORFLOW_DIR/include -I $CUDA_PATH/include -I $TENSORFLOW_DIR/include/external/nsync/public -lcudart -L $CUDA_PATH/lib64/ -L$TENSORFLOW_DIR/ -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=1
