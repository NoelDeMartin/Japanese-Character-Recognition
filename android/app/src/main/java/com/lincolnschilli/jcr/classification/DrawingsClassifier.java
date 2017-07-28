package com.lincolnschilli.jcr.classification;

import android.content.Context;

import com.lincolnschilli.jcr.drawing.Drawing;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;

public class DrawingsClassifier {

    private static int INPUT_WIDTH = 64;
    private static int INPUT_HEIGHT = 64;

    private static String INPUT_NAME = "input";
    private static String OUTPUT_NAME = "prediction";

    private HashMap<Integer, String> categories;
    private TensorFlowInferenceInterface tensorFlow;
    private float[] inputBuffer;
    private float[] outputBuffer;

    public DrawingsClassifier(Context context, String categoriesPath, String modelPath) throws IOException {
        loadCategories(context, categoriesPath);
        loadTensorFlowModel(context, modelPath);
    }

    public float classify(Drawing drawing) {

        drawing.drawBinaryData(inputBuffer, INPUT_WIDTH, INPUT_HEIGHT);

        this.tensorFlow.feed(INPUT_NAME, inputBuffer, 1, INPUT_WIDTH, INPUT_HEIGHT, 1);
        this.tensorFlow.run(new String[] { OUTPUT_NAME });
        this.tensorFlow.fetch(OUTPUT_NAME, outputBuffer);

        // TODO display this in the UI
        float maxValue = Integer.MIN_VALUE;
        int maxIndex = 0;
        for (int i = 0; i < outputBuffer.length; i++) {
            System.out.println(categories.get(i) + " --> " + outputBuffer[i]);
            if (outputBuffer[i] > maxValue) {
                maxValue = outputBuffer[i];
                maxIndex = i;
            }
        }

        System.out.println("CLASSIFICATION: " + categories.get(maxIndex));

        return maxIndex;
    }

    private void loadTensorFlowModel(Context context, String filePath) {
        this.tensorFlow = new TensorFlowInferenceInterface(context.getAssets(), filePath);
        inputBuffer = new float[INPUT_WIDTH * INPUT_HEIGHT];
        outputBuffer = new float[(int) tensorFlow.graphOperation(OUTPUT_NAME).output(0).shape().size(1)];
    }

    private void loadCategories(Context context, String filePath) throws IOException {
        InputStream inputStream = context.getAssets().open(filePath);
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));

            String line;

            // skip first line
            reader.readLine();

            categories = new HashMap<>();
            while ((line = reader.readLine()) != null) {
                String[] data = line.split(",");
                categories.put(Integer.parseInt(data[0]), data[1]);
            }
        } finally {
            inputStream.close();
        }
    }

}
