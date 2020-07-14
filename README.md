# Building for CountR Board

1. Build and compile MXNET

    a. Clone incubator-mxnet

    b. sudo apt-get install libopenblas-dev liblapack-dev libblas-dev libatlas-base-dev libjemalloc-dev libc6-dev-i386 ninja-build cmake

    c. cmake -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release -DUSE_OPENCV=0 -DUSE_F16C -DUSE_MKLDNN=0 -DBLAC=Atlas -GNinja ..

## Run Time Server

1. mvn clean && mvn package && mvn exec:java -PServer
2. Configurations are in src/main/resources/server.properties
    a. Change port to run multiple versions

## Downloading Pre-Trained model

Model is too large to have on GitHub. We must download and put it to the right location.
Here are some steps to do it. This isn't versioned and the file may become unavailable at some point. Contact dariuszlee@outlook.com if a backup version is required.

1. Go to link
https://github.com/deepinsight/insightface/wiki/Model-Zoo

2. Download Face Recognition: 3.3 LResNet34E-IR

3. Download Face Liveness model from here: https://drive.google.com/drive/folders/1AdfM_qC8NrxP1FuTyirNOU_cQ6IQlWXO?usp=sharing

## Dependencies

1. Opencv 4.2.0 - These needs to be installed WITH java support. You may need to copy the shared libraries around.
2. mtcnn-java - Check root directory readme for installation instructions.

## OpenCV 4.2.0 Build For Java  

Follow these instructions: 

1. https://opencv-java-tutorials.readthedocs.io/en/latest/01-installing-opencv-for-java.html
    a. Set build prefix to /usr/

2. Copy /usr/share/java/opencv4/libopencv_java{version}.so /usr/lib
    a. Version can is 420 in this case.
    b. If you can't find the shared library, run "find /usr | grep 420 | grep opencv"

## Cross Compatability

Needs Windows testing

## Known Issues

### Gstreamer Errors
1. Re-Build opencv and remove GStreamer support
    a. Open cmake-gui and search for gstreamer and unselect
