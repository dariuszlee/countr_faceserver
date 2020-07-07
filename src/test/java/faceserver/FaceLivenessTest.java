package faceserver;

import countr.common.ComputeUtils;
import countr.common.FaceDetection;
import countr.common.FaceLiveness;
import countr.common.MXNetUtils;
import countr.utils.DebugUtils;

import java.awt.image.BufferedImage;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import javax.imageio.ImageIO;
import junit.framework.Assert;
import org.junit.Test;

public class FaceLivenessTest {
  @Test
  public void LoadAndBasicTest() throws IOException {
    MXNetUtils predictor =
        new MXNetUtils(
            false,
            "/home/dzly/projects/countr_face_recognition/faceserver/src/main/resources/model-r100-ii/model");
    FaceLiveness livenessDetector =
        new FaceLiveness(
            false,
            "/home/dzly/projects/countr_face_recognition/faceserver/src/main/resources/faceliveness-model/anadfinal");

    InputStream imageInputStream =
        new FileInputStream(
            "/home/dzly/projects/countr_face_recognition/faceserver/src/main/resources/trainer_reference.jpg");
    BufferedImage inputImage = ImageIO.read(imageInputStream);

    FaceDetection faceDetector =
        new FaceDetection(false, inputImage.getWidth(), inputImage.getHeight());

    BufferedImage foundFace = faceDetector.detect(inputImage);
    DebugUtils.saveImage(foundFace, "LIVE2");
    boolean res = livenessDetector.isLive(foundFace);
    Assert.assertFalse(res);

    BufferedImage foundFace112 = faceDetector.resizeImage(foundFace, 112);
    float[] recognitionResults = predictor.predict(foundFace112);

    InputStream imageInputStream2 =
        new FileInputStream(
            "/home/dzly/projects/countr_face_recognition/faceserver/mpv-shot0002.jpg");
    BufferedImage inputImage2 = ImageIO.read(imageInputStream2);

    FaceDetection faceDetector2 =
        new FaceDetection(false, inputImage2.getWidth(), inputImage2.getHeight());

    BufferedImage foundFace2 = faceDetector2.detect(inputImage2);
    DebugUtils.saveImage(foundFace2, "LIVE");
    boolean res2 = livenessDetector.isLive(foundFace2);

    BufferedImage foundFace112_2 = faceDetector.resizeImage(foundFace2, 112);
    float[] recognitionResults2 = predictor.predict(foundFace112_2);

    float match = ComputeUtils.getDotProduct(recognitionResults, recognitionResults2);
    System.out.println("DARIUS " + match / recognitionResults.length);
  }
}
