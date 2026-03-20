# This is your make file
# You may change it and we use it to build your code.
# DO NOT CHANGE RECIPE FOR TEST RELATED TARGETS 
CXX := g++
NVCC := nvcc
INC_DIR := include
INCLUDES := -I$(INC_DIR) -I.
CXXFLAGS := -std=c++17 -Wall -Wextra -pedantic -MMD -MP
NVCCFLAGS := -std=c++17 $(INCLUDES) -Xcompiler="-Wall,-Wextra"
LDFLAGS := -lcudart
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
SOURCES := main.cpp $(SRC_DIR)/tokenizer_bpe.cpp $(SRC_DIR)/embedding.cpp $(SRC_DIR)/data_loader.cpp
OBJECTS := $(BUILD_DIR)/main.o $(BUILD_DIR)/tokenizer_bpe.o $(BUILD_DIR)/embedding.o $(BUILD_DIR)/data_loader.o $(BUILD_DIR)/kernels.o
DEPS := $(OBJECTS:.o=.d)

all: $(BIN_DIR)/$(TARGET)
$(BIN_DIR)/$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)
$(BUILD_DIR)/main.o: main.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/tokenizer_bpe.o: $(SRC_DIR)/tokenizer_bpe.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/embedding.o: $(SRC_DIR)/embedding.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/data_loader.o: $(SRC_DIR)/data_loader.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/kernels.o: kernel/kernels.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -MMD -MP -MF $(BUILD_DIR)/kernels.d -c $< -o $@
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

TEST_OBJECTS := $(BUILD_DIR)/test.o $(BUILD_DIR)/test_api.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/tokenizer_bpe.o $(BUILD_DIR)/embedding.o $(BUILD_DIR)/data_loader.o

tests: $(BIN_DIR)/tests

$(BIN_DIR)/tests: $(TEST_OBJECTS) | $(BIN_DIR)
	$(CXX) $(TEST_OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test.o: tests/test.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/test_api.o: tests/test_api.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
