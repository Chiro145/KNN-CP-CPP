CXX := g++
CXXFLAGS := -O3 -mavx512f -mavx512bw -mavx512dq -march=native -fopenmp -Wall -Wextra
TARGET := main
SRC := main.cpp
HEADERS := load_csv.h

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
