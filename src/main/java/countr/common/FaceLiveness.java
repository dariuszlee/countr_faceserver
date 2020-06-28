package countr.common;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.awt.image.BufferedImage;

import org.apache.mxnet.infer.javaapi.Predictor;
import org.apache.mxnet.javaapi.Context;
import org.apache.mxnet.javaapi.NDArray;
import org.apache.mxnet.javaapi.Shape;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;

public class FaceLiveness extends MXNetUtils {

    public FaceLiveness(boolean isGpu, String modelPath) {
      super(isGpu, modelPath);
    }        

    public void status(){
        System.out.println("Running...");
    }

    public boolean isLive(BufferedImage inputImage) throws IOException {
      Java2DNativeImageLoader imageLoader = new Java2DNativeImageLoader();
      INDArray ndImage3HW = imageLoader.asMatrix(inputImage).get(point(0), interval(0, 3), all(), all());
      // Need to divide by 255.0 for neural network
      ndImage3HW = ndImage3HW.div(255.0);

      float[] ints = ndImage3HW.data().asFloat();
      NDArray img = new NDArray(ints, this.inputShape, this.ctx.get(0));

      List<NDArray> imgs = new ArrayList<NDArray>();
      imgs.add(img);

      List<NDArray> res = this.resnet100.predictWithNDArray(imgs);
      System.out.println(res.get(0));
      return false;
    }
}