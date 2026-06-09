export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Create build directory
mkdir build && cd build
# cd build

# Configure the project

cmake DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc ..



# Build the project
cmake --build . --config Release

# Or use make on Unix systems
make -j$(nproc)

cp sph_cuda.cpython-3*.so ..
cd ..
rm -rf build