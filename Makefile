NVCC = nvcc
CFLAGS = -O3 -arch=sm_70
EXECUTABLES = vecAdd vecAdd_optimized

all: $(EXECUTABLES)

vecAdd: src/vecAdd.cu
    $(NVCC) $(CFLAGS) $< -o $@

vecAdd_optimized: src/vecAdd_optimized.cu
    $(NVCC) $(CFLAGS) $< -o $@

clean:
    rm -f $(EXECUTABLES)

.PHONY: all clean