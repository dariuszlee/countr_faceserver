package countr.common;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner;
import org.springframework.core.io.DefaultResourceLoader;

public class TensorFlowFaceRecognition {
	public static final String TF_PNET_MODEL_URI = "classpath:/tensor_model/best-m.pb";
	private final GraphRunner proposeNetGraphRunner;

    public TensorFlowFaceRecognition(){
        proposeNetGraphRunner = createGraphRunner(TF_PNET_MODEL_URI, "");
    }

    public INDArray run(INDArray image) {
        Map<String, INDArray> resultMap = proposeNetGraphRunner.run(Collections.singletonMap("", image));
        return resultMap.get("");
    }
    
	private GraphRunner createGraphRunner(String tensorflowModelUri, String inputLabel) {
		try {
			return new GraphRunner(
					IOUtils.toByteArray(new DefaultResourceLoader().getResource(tensorflowModelUri).getInputStream()),
					Arrays.asList(inputLabel));
					// ConfigProto.getDefaultInstance());
		}
		catch (IOException e) {
			throw new IllegalStateException(String.format("Failed to load TF model [%s] and input [%s]:",
					tensorflowModelUri, inputLabel), e);
		}
	}
}