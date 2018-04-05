package com.lincolnschilli.jcr.classification;

import android.content.Context;

import com.lincolnschilli.jcr.drawing.Drawing;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
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

    public Result[] classify(Drawing drawing) {

        drawing.drawBinaryData(inputBuffer, INPUT_WIDTH, INPUT_HEIGHT);

        this.tensorFlow.feed(INPUT_NAME, inputBuffer, 1, INPUT_WIDTH, INPUT_HEIGHT, 1);
        this.tensorFlow.run(new String[] { OUTPUT_NAME });
        this.tensorFlow.fetch(OUTPUT_NAME, outputBuffer);

        ArrayList<Result> results = new ArrayList<>();
        for (int i = 0; i < outputBuffer.length; i++) {
            results.add(new Result(categories.get(i), outputBuffer[i]));
        }

        Collections.sort(results, new Comparator<Result>() {
            @Override
            public int compare(Result a, Result b) {
                return a.getConfidence() > b.getConfidence()? -1 : 1;
            }
        });

        Result[] array = new Result[results.size()];
        results.toArray(array);

        return array;
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
