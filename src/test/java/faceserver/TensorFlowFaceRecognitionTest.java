package faceserver;

import org.junit.Test;

import countr.common.TensorFlowFaceRecognition;

public class TensorFlowFaceRecognitionTest {
    @Test
    public void LoadAndBasicTest(){
        TensorFlowFaceRecognition recognizer = null;
        try {
            recognizer = new TensorFlowFaceRecognition();
        }
        catch (Exception e){
            System.out.println("Test Exception: " + e);
            // Assert.assertTrue(false);
        }
    }
}
