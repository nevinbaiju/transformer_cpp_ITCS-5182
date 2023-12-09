CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -std=c++11 
AVX_FLAGS = -fopenmp -mavx -march=native -mtune=native

SRCS = main.cpp activations.cpp initializations.cpp helpers.cpp computations.cpp exceptions.cpp attention.cpp Tensor.cpp
CUDA_SRCS = cuda_kernels.cu D_Tensor.cu d_activations.cu

NORMAL_OBJS = $(SRCS:.cpp=_normal.o)
EXEC_NORMAL = run_transformer_normal

AVX_OBJS = $(SRCS:.cpp=_avx.o)
EXEC_AVX = run_transformer_avx

CUDA_CU_OBJS = $(CUDA_SRCS:.cu=_cuda.o)
CUDA_OBJS = $(SRCS:.cpp=_cuda.o)
EXEC_CUDA = run_transformer_cuda

$(info The value of SOME_VARIABLE is: $(CUDA_SRCS))

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

# CUDA Compilation
$(EXEC_CUDA): $(CUDA_OBJS) $(CUDA_CU_OBJS)
	$(NVCC) $(CXXFLAGS) -DCUDA $^ -o $@
$(CUDA_CU_OBJS): %_cuda.o: %.cu
	$(NVCC) $(CXXFLAGS) -DCUDA -c $< -o $@
$(CUDA_OBJS): %_cuda.o: %.cpp
	$(NVCC) $(CXXFLAGS) -DCUDA -c $< -o $@

clean:
	rm -f *.o