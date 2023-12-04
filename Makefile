CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -std=c++11 
AVX_FLAGS = -fopenmp -mavx -march=native -mtune=native

SRCS = main.cpp activations.cpp initializations.cpp helpers.cpp computations.cpp exceptions.cpp attention.cpp Tensor.cpp
CUDA_SRCS = cuda_kernels.cu

NORMAL_OBJS = $(SRCS:.cpp=_normal.o)
EXEC_NORMAL = run_transformer_normal

AVX_OBJS = $(SRCS:.cpp=_avx.o)
EXEC_AVX = run_transformer_avx

CUDA_OBJS = $(SRCS:.cpp=_cuda.o)
EXEC_CUDA = run_transformer_cuda

.PHONY: all clean

all: $(EXEC_NORMAL) $(EXEC_AVX) $(EXEC_CUDA)

#Normal (Gold standard) compilation
$(EXEC_NORMAL): $(NORMAL_OBJS)
	$(CXX) $(CXXFLAGS)  -DNORMAL $(NORMAL_OBJS) -o $(EXEC_NORMAL)
%_normal.o: %.cpp
	$(CXX) $(CXXFLAGS) -DNORMAL -c $< -o $@

#AVX compilation
$(EXEC_AVX): $(AVX_OBJS)
	$(CXX) $(CXXFLAGS) $(AVX_FLAGS) -DAVX $(AVX_OBJS) -o $(EXEC_AVX)
%_avx.o: %.cpp
	$(CXX) $(CXXFLAGS) $(AVX_FLAGS) -DAVX -c $< -o $@

#Cuda compilation
$(EXEC_CUDA): $(CUDA_OBJS)
	$(NVCC) $(CXXFLAGS) -DCUDA $(CUDA_OBJS) -o $(EXEC_CUDA)
	$(NVCC) $(CXXFLAGS) -DCUDA cuda_kernels.cu -o cuda_kernels.o
%_cuda.o: %.cpp
	$(NVCC) $(CXXFLAGS) -DCUDA  -c $< -o $@

clean:
	rm -f $(NORMAL_OBJS) $(AVX_OBJS) $(EXEC_NORMAL) $(EXEC_AVX)
