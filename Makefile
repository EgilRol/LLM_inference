# This is your make file
# You may change it and we use it to build your code.
# DO NOT CHANGE RECIPE FOR TEST RELATED TARGETS 
CXX := g++
NVCC := nvcc
# Driver-only installs (nvidia-smi) do not ship headers; the toolkit does.
# Set CUDA_HOME if headers live elsewhere (e.g. module-loaded CUDA).
CUDA_HOME ?= /usr/local/cuda
INC_DIR := include
INCLUDES := -I$(INC_DIR) -I. -I$(CUDA_HOME)/include
CXXFLAGS := -std=c++17 -Wall -Wextra -pedantic -MMD -MP
NVCCFLAGS := -std=c++17 $(INCLUDES) -Xcompiler="-Wall,-Wextra"
LDFLAGS := -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart
BUILD ?= release

# Target modern NVIDIA GPUs (e.g. A100, H100). Use sm_70 for V100 or older.
NVCC_ARCH ?= -arch=sm_80

ifeq ($(BUILD),debug)
  CXXFLAGS += -g -O0
  NVCCFLAGS += -g -G -O0
else
  CXXFLAGS += -O2
  NVCCFLAGS += -O3 $(NVCC_ARCH)
endif

SRC_DIR := src
BUILD_DIR := build
BIN_DIR := bin
TARGET := llm
CPP_SOURCES := \
	main.cpp \
	$(SRC_DIR)/tokenizer_bpe.cpp \
	$(SRC_DIR)/io/weight_index.cpp \
	$(SRC_DIR)/io/weight_loader.cpp \
	$(SRC_DIR)/io/staged_reader.cpp \
	$(SRC_DIR)/operators/embedding_op.cpp \
	$(SRC_DIR)/operators/linear_op.cpp \
	$(SRC_DIR)/operators/matmul_op.cpp \
	$(SRC_DIR)/operators/rmsnorm_op.cpp \
	$(SRC_DIR)/runtime/cuda_context.cpp \
	$(SRC_DIR)/runtime/workspace.cpp \
	$(SRC_DIR)/runtime/model_weights_gpu.cpp
CUDA_SOURCES := \
	kernel/matmul.cu \
	kernel/rmsnorm.cu \
	kernel/embedding.cu
CPP_OBJECTS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(CPP_SOURCES))
CUDA_OBJECTS := $(patsubst %.cu,$(BUILD_DIR)/%.o,$(CUDA_SOURCES))
OBJECTS := $(CPP_OBJECTS) $(CUDA_OBJECTS)
DEPS := $(OBJECTS:.o=.d)

all: $(BIN_DIR)/$(TARGET)
$(BIN_DIR)/$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: %.cu | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

.PHONY: clean

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	rm -f a.out

.PHONY: run

run: all
	./$(BIN_DIR)/$(TARGET)
-include $(DEPS)

# ------------------------------------------------------------
# Tests build
# Don't forget to add your required objects as well.

.PHONY: tests

CORE_OBJECTS := $(filter-out $(BUILD_DIR)/main.o,$(CPP_OBJECTS)) $(CUDA_OBJECTS)
TEST_OBJECTS := $(BUILD_DIR)/tests/test.o $(BUILD_DIR)/tests/test_api.o $(CORE_OBJECTS)

tests: $(BIN_DIR)/tests

$(BIN_DIR)/tests: $(TEST_OBJECTS) | $(BIN_DIR)
	$(CXX) $(TEST_OBJECTS) -o $@ $(LDFLAGS)
