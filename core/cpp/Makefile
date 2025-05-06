# Project Nanotron — C++ Build Configuration
# Optimized for ultra-low latency

CXX := g++
CXXFLAGS := -std=c++23 -O3 -march=native -mtune=native -flto
CXXFLAGS += -Wall -Wextra -Wpedantic -Werror
CXXFLAGS += -fno-exceptions -fno-rtti  # Disable for minimal binary
CXXFLAGS += -ffast-math -funroll-loops
CXXFLAGS += -DNDEBUG

# Profile-guided optimization
PGO_GEN := -fprofile-generate
PGO_USE := -fprofile-use -fprofile-correction

# Link flags
LDFLAGS := -lpthread -lrt

# Formal verification with Frama-C
FRAMA_C := frama-c
FRAMA_FLAGS := -wp -wp-rte -wp-timeout 60

# Sources
SRCS := order_gateway.cpp
OBJS := $(SRCS:.cpp=.o)
TARGET := nanotron_executor

# Headers
HDRS := shared_memory.hpp

.PHONY: all clean verify pgo

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Profile-guided optimization build
pgo:
	# Step 1: Build with instrumentation
	$(CXX) $(CXXFLAGS) $(PGO_GEN) -o $(TARGET)_pgo_gen $(SRCS) $(LDFLAGS)
	
	# Step 2: Run with representative workload
	./$(TARGET)_pgo_gen --benchmark || true
	
	# Step 3: Build with profile data
	$(CXX) $(CXXFLAGS) $(PGO_USE) -o $(TARGET) $(SRCS) $(LDFLAGS)
	
	# Cleanup
	rm -f $(TARGET)_pgo_gen *.gcda

# Formal verification
verify:
	$(FRAMA_C) $(FRAMA_FLAGS) order_gateway.cpp

# Static analysis
analyze:
	clang-tidy $(SRCS) -- $(CXXFLAGS)
	cppcheck --enable=all --std=c++23 $(SRCS)

# Benchmark
benchmark: $(TARGET)
	perf stat -d ./$(TARGET) --benchmark

clean:
	rm -f $(OBJS) $(TARGET) $(TARGET)_pgo_gen *.gcda *.gcno

# Install
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

# Debug build
debug: CXXFLAGS := -std=c++23 -g -O0 -DDEBUG
debug: $(TARGET)

# Sanitizer build
sanitize: CXXFLAGS += -fsanitize=address,undefined
sanitize: LDFLAGS += -fsanitize=address,undefined
sanitize: $(TARGET)

