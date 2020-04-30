package faceserver;

import static org.nd4j.linalg.indexing.NDArrayIndex.point;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.datavec.image.loader.Java2DNativeImageLoader;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.opencv.core.Core;

import countr.common.FaceDetection;
import countr.common.FaceLiveness;


public class FaceLivenessTest {
    static {
        System.out.println("LIBRARY: " + Core.NATIVE_LIBRARY_NAME);
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private INDArray real;
    private INDArray fake;
    private FaceLiveness livenessDetector;
    private FaceDetection faceDetector;
    private Java2DNativeImageLoader imageLoader;

    public FaceLivenessTest() throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        livenessDetector = new FaceLiveness();
        faceDetector = new FaceDetection(true);
        this.imageLoader = new Java2DNativeImageLoader();
    }

    @org.junit.Before
    public void Before() throws IOException {
    }

    private INDArrayIndex interval(int i, int j) {
		return null;
	}

	@Test
    public void LoadAndBasicTest() throws IOException, Exception {
        // String filePath = "file:/home/dzly/projects/countr_face_recognition/faceserver/nuaa/raw/ClientRaw/0001/0001_00_00_02_0.jpg";
        String filePath = "/home/dzly/projects/countr_face_recognition/faceserver/nuaa/raw/ClientRaw";
        // Files.walk(Paths.get(filePath))
        //     .filter(Files::isRegularFile)
        //     .forEach(filePath2 -> {
        //         try {
        //             System.out.println(filePath2);
        //             File tempFile = filePath2.toFile();
        //             InputStream targetStream = new FileInputStream(tempFile);

        //             INDArray ndImage3HW = this.imageLoader.asMatrix(targetStream).reshape(new int[]{3, 480, 640});
        //             BufferedImage image = faceDetector.detect(ndImage3HW, 128);

        //             INDArray detectedImageArray = imageLoader.asMatrix(image);
        //             detectedImageArray = detectedImageArray.reshape(new int[]{1, 128, 128, 3});
        //             INDArray out = livenessDetector.output(detectedImageArray);
        //             System.out.println("DARIUS 2 " + out.get(point(0)).getFloat(0));
        //             float real =  out.get(point(0)).getFloat(1);
        //             System.out.println("DARIUS 3 " + real);
        //             // if(real == 1){
        //             //     System.exit(1);
        //             // }
        //             // System.out.println("DARIUS " + livenessDetector.predict(detectedImageArray));
        //             // Arrays.stream(livenessDetector.predict(detectedImageArray)).forEach(x -> System.out.println("DARIUS " + x));
        //         }
        //         catch (Exception e){
        //             System.out.println("Ex: "+ e);
        //         }
        //     });
    }
}