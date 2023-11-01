CXX = g++
CXXFLAGS = -std=c++11

SRCS = main.cpp activations.cpp initializations.cpp helpers.cpp computations.cpp exceptions.cpp
OBJS = $(SRCS:.cpp=.o)
EXEC = run_transformer

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(EXEC)

activations.o: activations.cpp activations.h
	$(CXX) $(CXXFLAGS) -c activations.cpp

main.o: main.cpp activations.h
	$(CXX) $(CXXFLAGS) -c main.cpp

run:
	./run_transformer

clean:
	rm -f $(OBJS) $(EXEC)
