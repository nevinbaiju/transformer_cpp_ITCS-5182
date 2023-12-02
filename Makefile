CXX = g++
CXXFLAGS = -std=c++11 -O3
AVX_FLAGS = -fopenmp -mavx -march=native -mtune=native

SRCS = main.cpp activations.cpp initializations.cpp helpers.cpp computations.cpp exceptions.cpp attention.cpp Tensor.cpp

NORMAL_OBJS = $(SRCS:.cpp=.o)
EXEC_NORMAL = run_transformer

AVX_OBJS = $(SRCS:.cpp=_avx.o)
EXEC_AVX = run_transformer_avx

.PHONY: all clean

all: $(EXEC_NORMAL)

$(EXEC_NORMAL): $(NORMAL_OBJS)
	$(CXX) $(CXXFLAGS) -DNORMAL $(NORMAL_OBJS) -o $(EXEC_NORMAL)

$(EXEC_AVX): $(AVX_OBJS)
	$(CXX) $(CXXFLAGS) $(AVX_FLAGS) -DAVX $(AVX_OBJS) -o $(EXEC_AVX)


%.o: %.cpp
	$(CXX) $(CXXFLAGS) -DNORMAL -c $< -o $@

%_avx.o: %.cpp
	$(CXX) $(CXXFLAGS) $(AVX_FLAGS) -DAVX -c $< -o $@

clean:
	rm -f $(NORMAL_OBJS) $(AVX_OBJS) $(EXEC_NORMAL) $(EXEC_AVX)
