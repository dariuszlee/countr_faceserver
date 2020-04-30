package countr.common;

import org.apache.commons.io.IOUtils;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.DefaultResourceLoader;

public class FaceLiveness {
    // private final MultiLayerNetwork model;
	// private final GraphRunner proposeNetGraphRunner;

    public FaceLiveness() throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        // String path = "../Face-Liveness-Detection/models/anandfinal.h5";
        // String pathJson = "../Face-Liveness-Detection/models/anandfinal.json";
        // String fullPath = "classpath:jFace-Liveness-Detection/models/anandfinal.hdf5";
        // fullPath = new ClassPathResource("anandfinal.hdf5", this.getClass().getClassLoader()).getURI().getPath();
        // // model = KerasModelImport.importKerasSequentialModelAndWeights(pathJson, path);
        // // fullPath = "/home/dzly/Downloads/liveness-detection-opencv/anandfinal.hdf5";
        // model = KerasModelImport.importKerasSequentialModelAndWeights(fullPath);
        // model.init();

        // String inputNode = "separable_conv2d_16/separable_conv2d/depthwise";
        // fullPath = new ClassPathResource("anandfinal.pb", this.getClass().getClassLoader()).getURI().getPath();
        // String inputNode = "separable_conv2d_16_input";
        // this.proposeNetGraphRunner = this.createGraphRunner(fullPath, inputNode, Stream.of("activation_40/Sigmoid").collect(Collectors.toList()));

    }        
    
	// private GraphRunner createGraphRunner(String tensorflowModelUri, String inputLabel, List<String> outputLabels) {
	// 	try {
            // File file = new File(tensorflowModelUri);
            // InputStream targetStream = new FileInputStream(file);
	// 		return GraphRunner.builder()
                // .graphBytes(IOUtils.toByteArray(targetStream))
                // .inputNames(Arrays.asList(inputLabel))
                // .outputNames(outputLabels)
                // .build();
	// 				// ConfigProto.getDefaultInstance());
	// 	}
	// 	catch (IOException e) {
	// 		throw new IllegalStateException(String.format("Failed to load TF model [%s] and input [%s]:",
	// 				tensorflowModelUri, inputLabel), e);
	// 	}
	// }

    // public int[] predict(INDArray toPredict) {
        // return model.predict(toPredict);
    // }
    
    // public INDArray output(INDArray toPredict) {
        // System.out.println("DARIUS SHAPE " + toPredict.shapeInfoToString());
        // Map<String, INDArray> inputMap = new HashMap<>();
        // inputMap.put("separable_conv2d_16_input", toPredict);
        // Map<String, INDArray> outputMap = this.proposeNetGraphRunner.run(inputMap);
        // INDArray output = outputMap.get("activation_40/Sigmoid");
        // System.out.println("DARIUS OUT " + output);
        // return output;
        // // return model.output(toPredict);
    // }

    // public void status(){
        // System.out.println("Running...");
        // System.out.println();
    // }

    // private File createTempFile(String prefix, String suffix) throws IOException {
        // return File.createTempFile("temp", "asdf");
    // }
}