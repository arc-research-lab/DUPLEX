# ================== DUPLEX build ================== #
# Usage:

# ---- platform paths (override accordingly) ---- #
PLATFORM ?= /tools/Xilinx/Vitis/2024.2/base_platforms/xilinx_vck190_base_202420_1/xilinx_vck190_base_20240_1.xpfm
COMMON ?= $(HOME)/platform/xilinx-versal-common-v2024.2
TARGET ?= hw

# ---- design ---- # 
KERNEL ?= top 
KERN_SRCS := src/top.cpp 
HOST_SRCS := src/host_top.cpp 

# ---- configs ---- # 
M  ?= 16
NK ?= 16
NW ?= 1
D  ?= 8
CFG   := m$(M)_nk$(NK)_nw$(NW)_d$(D)
BUILD := build/$(TARGET)/$(CFG)
DEFS  := -D M=$(M) -D N_K=$(NK) -D N_W=$(NW) -D DEPTH=$(D) 

.PHONY: xo xsa package host clean 
xo: $(BUILD)/$(KERNEL).xo
xsa: $(BUILD)/$(KERNEL).xsa
package: $(BUILD)/package/sd_card.img 

$(BUILD)/$(KERNEL).xo: $(KERN_SRCS)
	mkdir -p $(BUILD)
	v++ -c -t $(TARGET) --platform $(PLATFORM) -k $(KERNEL) $(DEFS) -I src \ 
		--temp_dir $(BUILD)/_x -o $@ $^ 

$(BUILD)/$(KERNEL).xsa: $(BUILD)/$(KERNEL).xo
	v++ -l -t $(TARGET) --platform $(PLATFORM) \ 
		--temp_dir $(BUILD)/_x -o $@ $< 

$(BUILD)/package/sd_card.img: $(BUILD)/$(KERNEL).xsa build/host.exe 
	v++ -p -t $(TARGET) --platform $(PLATFORM) $< \ 
		--package.out_dir $(BUILD)/package \ 
		--package.boot_mode sd \ 
		--package.rootfs $(COMMON)/rootfs.ext4 \ 
		--package.kernel_image $(COMMON)/Image \ 
		--package.sd_file build/host.exe \ 
		-o $(BUILD)/$(KERNEL).xclbin 

build/host.exe: $(HOST_SRCS)
	@case "$(CXX)" in *aarch64*) ;; *) echo "Source the Versal SDK environment first"; exit 1;; esac
	mkdir -p build
	$(CXX) -std=c++17 $^ -o $@ -I$(SDKTARGETSYSROOT)/usr/include/xrt -lxrt_coreutil -pthread

host: build/host.exe

clean:
	rm -rf build .Xil *.log