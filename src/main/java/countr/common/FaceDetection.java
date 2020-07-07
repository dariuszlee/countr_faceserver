package countr.common;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

import countr.utils.DebugUtils;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import net.tzolov.cv.mtcnn.FaceAnnotation;
import net.tzolov.cv.mtcnn.FaceAnnotation.BoundingBox;
import net.tzolov.cv.mtcnn.MtcnnService;
import net.tzolov.cv.mtcnn.MtcnnUtil;
import org.bytedeco.javacpp.opencv_core;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

public class FaceDetection {
  MtcnnService mtcnnService;
  boolean isDebug;

  public FaceDetection(boolean isDebug, int imageWidth, int imageHeight) throws IOException {
    this.mtcnnService =
        new MtcnnService(30, 0.709, new double[] {0.6, 0.7, 0.8}, imageWidth, imageHeight);
    this.isDebug = isDebug;
  }

  public BufferedImage detect(BufferedImage inputImage) throws IOException {
    // 2. Run face detection
    Java2DNativeImageLoader imageLoader = new Java2DNativeImageLoader();
    INDArray ndImage3HW =
        imageLoader.asMatrix(inputImage).get(point(0), interval(0, 3), all(), all());
    FaceAnnotation[] faceAnnotations = this.mtcnnService.faceDetection(ndImage3HW);

    System.out.println("Face length: " + faceAnnotations.length);
    if (faceAnnotations.length == 0) {
      DebugUtils.saveImage(inputImage, "received_after");
      return null;
    }

    if (this.isDebug) {
      BufferedImage annotatedImage = MtcnnUtil.drawFaceAnnotations(inputImage, faceAnnotations);
      ClassLoader classloader = Thread.currentThread().getContextClassLoader();
      String directoryPath = classloader.getResource("debug/").getPath();
      String imageDebugPath = directoryPath + "./AnnotatedImage.png";
      ImageIO.write(annotatedImage, "png", new File(imageDebugPath));
    }

    int margin = 44;
    int alignedImageSize = 128;
    FaceAnnotation bbox = this.getBiggestFace(faceAnnotations);
    INDArray alignedFace =
        mtcnnService.faceAlignment(ndImage3HW, bbox, margin, alignedImageSize, false);
    BufferedImage image = new Java2DNativeImageLoader().asBufferedImage(alignedFace);

    return image;
  }

  public BufferedImage resizeImage(BufferedImage toResize, int width) throws IOException {
    INDArray ndImage3HW =
        new Java2DNativeImageLoader().asMatrix(toResize).get(point(0), interval(0, 3), all(), all());

    INDArray resized = mtcnnService.resize(ndImage3HW, new opencv_core.Size(width, width));
    BufferedImage image = new Java2DNativeImageLoader().asBufferedImage(resized);
    return image;
  }

  private FaceAnnotation getBiggestFace(FaceAnnotation[] faces) {
    int largestArea = 0;
    FaceAnnotation largestAreaIndex = null;
    for (FaceAnnotation face : faces) {
      BoundingBox bbox = face.getBoundingBox();
      int area = bbox.getW() * bbox.getH();
      if (area > largestArea) {
        largestAreaIndex = face;
        largestArea = area;
      }
    }
    return largestAreaIndex;
  }
}
