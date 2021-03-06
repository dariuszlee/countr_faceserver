FROM ubuntu:bionic

RUN apt-get update -y
RUN apt-get install openjdk-8-jdk -y
RUN apt-get install git -y
RUN apt-get install maven -y

RUN git clone https://github.com/dariuszlee/countr_facecommon
RUN git clone https://github.com/dariuszlee/mtcnn-java

COPY ./src/ ./faceserver/src
COPY pom.xml ./faceserver/

WORKDIR ./countr_facecommon/
RUN mvn clean install


WORKDIR ../mtcnn-java/
RUN mvn clean install
WORKDIR ../

# WORKDIR ../
# RUN mvn clean package

RUN apt-get install -y less curl unzip cmake build-essential python
RUN curl https://codeload.github.com/opencv/opencv/zip/3.4.2 -o opencv342.zip
RUN unzip opencv342.zip
WORKDIR ./opencv-3.4.2/
RUN mkdir build
RUN apt-get install -y ant
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64
WORKDIR build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_OPENCL=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_SHARED_LIBS=OFF -D JAVA_INCLUDE_PATH=$JAVA_HOME/include -D JAVA_AWT_LIBRARY=$JAVA_HOME/jre/lib/arm/libawt.so -D JAVA_JVM_LIBRARY=$JAVA_HOME/jre/lib/arm/server/libjvm.so -D CMAKE_INSTALL_PREFIX=/usr ..
RUN make -j10
RUN make install
RUN cp /usr/share/OpenCV/java/libopencv_java342.so /usr/lib/
WORKDIR ../../faceserver
RUN mvn clean package

CMD mvn exec:java -Dexec.mainClass="countr.faceserver.FaceServer"
