# ====================================================================
# Unified Makefile for arm64 and x64
#
# Usage:
#   make              - Builds for the current system architecture (auto-detected).
#   make ARCH=arm64   - Forces a build for arm64.
#   make ARCH=x64     - Forces a build for x64.
#   make clean        - Removes build artifacts.
# ====================================================================

# --- 1. Common Variables ---
ROOT_DIR := .
OBJ_DIR  := obj
SRC_DIR  := src

# Define all targets
TARGET_DETECT   := pkshin_detect
TARGET_MAP_COCO := cal_mAP_coco
TARGET_MAP_DOTA := cal_mAP_dota

CXX := g++ -std=c++17 -O2

# Source file list for the detection engine
SRCS_DETECT := $(SRC_DIR)/main.cpp \
               $(SRC_DIR)/run_image.cpp \
               $(SRC_DIR)/run_camera.cpp \
               $(SRC_DIR)/run_video.cpp \
               $(SRC_DIR)/run_demo.cpp \
               $(SRC_DIR)/run_multicam.cpp

# Source file list for the mAP calculators
SRCS_MAP_COCO := $(SRC_DIR)/cal_mAP_coco.cpp
SRCS_MAP_DOTA := $(SRC_DIR)/cal_mAP_dota.cpp

# --- 2. Architecture Detection and Configuration ---
ifeq ($(ARCH),)
    UNAME_M := $(shell uname -m)
    ifeq ($(UNAME_M), aarch64)
        ARCH := arm64
    else
        ARCH := x64
    endif
endif

ifeq ($(ARCH), arm64)
    # --- ARM64 Configuration ---
    $(info Building for ARM64 architecture...)
    INC_DIR := $(ROOT_DIR)/include_arm64
    LIB_DIR := $(ROOT_DIR)/lib_arm64

    NEW_LIB_NAME := libqbruntime.so
    FOUND_IN_CUSTOM := $(wildcard $(LIB_DIR)/$(NEW_LIB_NAME))

    FOUND_IN_LDCONFIG := $(shell ldconfig -p | grep $(NEW_LIB_NAME))

    # Set Flags based on detection
    ifneq ($(FOUND_IN_CUSTOM),)
        $(info [Config] Found $(NEW_LIB_NAME) in custom directory: $(LIB_DIR))
        MACCEL_LIBS := -lqbruntime
        LIB_DEFINES := -DUSE_QBRUNTIME_LIB
    else ifneq ($(FOUND_IN_LDCONFIG),)
        $(info [Config] Found $(NEW_LIB_NAME) via ldconfig (system cache))
        MACCEL_LIBS := -lqbruntime
        LIB_DEFINES := -DUSE_QBRUNTIME_LIB
    else
        $(info [Config] $(NEW_LIB_NAME) not found. Falling back to libmaccel.so)
        MACCEL_LIBS := -lmaccel
        LIB_DEFINES := -DUSE_OLD_MACCEL_LIB
    endif

    CXXFLAGS := -mcpu=cortex-a76 -fopenmp $(LIB_DEFINES)
    LDFLAGS := -Wl,-rpath,$(LIB_DIR) \
               -lpkshin_engine \
               -ltensorflowlite -ltensorflowlite_gpu_delegate -ltensorflowlite_hexagon_delegate \
               $(MACCEL_LIBS) \
               -lhailort \
               -lais_client -lfastcvopt -lqcarcam_client \
               -ljson-c -ljpeg \
               -lgstreamer-1.0 -lgstapp-1.0 -lgobject-2.0 -lglib-2.0 -lgstvideo-1.0 \
               -lwayland-client -lwayland-egl -lEGL -lGLESv2 \
               -lopencv_core -lopencv_imgproc -lopencv_dnn -lopencv_imgcodecs -lopencv_calib3d

    INCS := -I$(INC_DIR)
    LIBS := -L$(LIB_DIR)

else ifeq ($(ARCH), x64)
    # --- x64 Configuration ---
    $(info Building for x64 architecture...)
    INC_DIR := $(ROOT_DIR)/include_x64
    LIB_DIR := $(ROOT_DIR)/lib_x64

    NEW_LIB_NAME := libqbruntime.so
    FOUND_IN_CUSTOM := $(wildcard $(LIB_DIR)/$(NEW_LIB_NAME))

    FOUND_IN_LDCONFIG := $(shell ldconfig -p | grep $(NEW_LIB_NAME))

    # Set Flags based on detection
    ifneq ($(FOUND_IN_CUSTOM),)
        $(info [Config] Found $(NEW_LIB_NAME) in custom directory: $(LIB_DIR))
        MACCEL_LIBS := -lqbruntime
        LIB_DEFINES := -DUSE_QBRUNTIME_LIB
    else ifneq ($(FOUND_IN_LDCONFIG),)
        $(info [Config] Found $(NEW_LIB_NAME) via ldconfig (system cache))
        MACCEL_LIBS := -lqbruntime
        LIB_DEFINES := -DUSE_QBRUNTIME_LIB
    else
        $(info [Config] $(NEW_LIB_NAME) not found. Falling back to libmaccel.so)
        MACCEL_LIBS := -lmaccel
        LIB_DEFINES := -DUSE_OLD_MACCEL_LIB
    endif

    CXXFLAGS := -fopenmp $(LIB_DEFINES)
    LDFLAGS := -Wl,-rpath,$(LIB_DIR) \
               -lpkshin_engine \
               -ltensorflowlite -ltensorflowlite_gpu_delegate -ltensorflowlite_hexagon_delegate \
               $(MACCEL_LIBS) \
               -lhailort \
               -ljson-c -ljpeg \
               -lwayland-client -lwayland-egl -lEGL -lGLESv2

    INCS := -I$(INC_DIR) $(shell pkg-config --cflags opencv4 gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
    LIBS := -L$(LIB_DIR) $(shell pkg-config --libs opencv4 gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)

else
    $(error Unsupported architecture: $(ARCH). Use 'arm64' or 'x64'.)
endif


# --- 3. Build Rules (Common for all architectures) ---
HDRS := $(shell find $(SRC_DIR) -name '*.hpp')

OBJS_DETECT   := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS_DETECT))
OBJS_MAP_COCO := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS_MAP_COCO))
OBJS_MAP_DOTA := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS_MAP_DOTA))

DEPS := $(OBJS_DETECT:.o=.d) $(OBJS_MAP_COCO:.o=.d) $(OBJS_MAP_DOTA:.o=.d)

.PHONY: all clean

# Default rule builds all executables
all: $(TARGET_DETECT) $(TARGET_MAP_COCO) $(TARGET_MAP_DOTA)
	@echo "Build successfully for $(ARCH)."
	@echo "Executables created: $(TARGET_DETECT), $(TARGET_MAP_COCO), and $(TARGET_MAP_DOTA)."

# Rule to compile source files into object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HDRS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCS) -c $< -o $@ -MD

# Rule to link the main detection engine
$(TARGET_DETECT): $(OBJS_DETECT)
	$(CXX) $(CXXFLAGS) $(OBJS_DETECT) -o $@ $(LIBS) $(LDFLAGS)

# Rule to link the COCO mAP calculator
$(TARGET_MAP_COCO): $(OBJS_MAP_COCO)
	$(CXX) $(CXXFLAGS) $(OBJS_MAP_COCO) -o $@ $(LIBS) $(LDFLAGS)

# Rule to link the DOTA mAP calculator
$(TARGET_MAP_DOTA): $(OBJS_MAP_DOTA)
	$(CXX) $(CXXFLAGS) $(OBJS_MAP_DOTA) -o $@ $(LIBS) $(LDFLAGS)

clean:
	@echo "Cleaning up..."
	rm -f $(TARGET_DETECT) $(TARGET_MAP_COCO) $(TARGET_MAP_DOTA)
	rm -rf $(OBJ_DIR)

# Include dependency files generated by the compiler
-include $(DEPS)