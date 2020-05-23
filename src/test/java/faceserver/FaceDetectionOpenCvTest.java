package faceserver;

import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.ResourceLoader;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.junit.Test;
import org.opencv.core.Core;

import countr.facedetection.FaceDetectionOpenCv;
import junit.framework.Assert;

public class FaceDetectionOpenCvTest {
    FaceDetectionOpenCv detector;

    public FaceDetectionOpenCvTest() {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        try {
            detector = new FaceDetectionOpenCv(128);
        }
        catch (Exception e){
            System.out.println("Test Exception: " + e);
        }
    }
    @Test
    public void LoadAndBasicTest(){
        BufferedImage img = null;
		ResourceLoader resourceLoader = new DefaultResourceLoader();
        try {
            File path = resourceLoader.getResource("classpath:/Detlef_test.png").getFile();
            img = ImageIO.read(path);
        } catch (IOException e) {
            System.out.println("Can't read image" + e);
        }

        BufferedImage foundImage = detector.detect(img);
        try{
            ImageIO.write(foundImage, "png", new File("./test.png"));
        } catch (IOException e) {
            System.out.println("Can't read image" + e);
        }
    }
}
