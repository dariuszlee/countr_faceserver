package countr.facedetection;

import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import java.awt.image.BufferedImage;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Size;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.ResourceLoader;
import org.opencv.imgproc.Imgproc;

public class FaceDetectionOpenCv {
    private CascadeClassifier faceCascade;
    private int size;

    public FaceDetectionOpenCv(int size) throws IOException {
        this.faceCascade = new CascadeClassifier();
		ResourceLoader resourceLoader = new DefaultResourceLoader();
        String path = resourceLoader.getResource("classpath:/haar_cascade.xml").getFile().getAbsolutePath();
        this.faceCascade.load(path);
        this.size = size;
    }    

    public BufferedImage detect(BufferedImage inputImage){
        Mat inputImageMat = bufferedImageToMat(inputImage);
        MatOfRect faces = new MatOfRect();

        faceCascade.detectMultiScale(inputImageMat, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE, new Size(this.size, this.size), new Size());

        if(faces.empty()){
            return null;
        }

        Mat cropped = new Mat(this.size, this.size, 0);
        Imgproc.resize(new Mat(inputImageMat, faces.toArray()[0]), cropped, new Size(this.size, this.size));

        BufferedImage outputImage = matToBufferedImage(cropped);
        return outputImage;
    }
            
    public static Mat bufferedImageToMat(BufferedImage bi) {
        Mat mat = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC3);
        byte[] data = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
        mat.put(0, 0, data);
        return mat;
    }

    private static BufferedImage matToBufferedImage(Mat original)
	{
		// init
		BufferedImage image = null;
		int width = original.width(), height = original.height(), channels = original.channels();
		byte[] sourcePixels = new byte[width * height * channels];
		original.get(0, 0, sourcePixels);
		
		if (original.channels() > 1)
		{
			image = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
		}
		else
		{
			image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
		}
		final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
		System.arraycopy(sourcePixels, 0, targetPixels, 0, sourcePixels.length);
		
		return image;
	}
}