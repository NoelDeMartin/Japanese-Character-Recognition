package com.lincolnschilli.jcr;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import com.lincolnschilli.jcr.classification.DrawingsClassifier;
import com.lincolnschilli.jcr.classification.Result;
import com.lincolnschilli.jcr.classification.tasks.LoadModelTask;
import com.lincolnschilli.jcr.classification.tasks.LoadModelTaskListener;
import com.lincolnschilli.jcr.views.DrawingSheetView;

public class MainActivity extends AppCompatActivity implements LoadModelTaskListener {

    private DrawingsClassifier classifier;

    private DrawingSheetView canvas;
    private ImageView preview;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        LoadModelTask task = new LoadModelTask(this, this);
        task.execute("categories.csv", "model.pb");

        canvas = ((DrawingSheetView) findViewById(R.id.canvas));
        preview = ((ImageView) findViewById(R.id.preview));

        findViewById(R.id.classify).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (classifier != null) {
                    preview.setImageBitmap(canvas.getDrawing().getBitmap(64, 64));
                    Result result = classifier.classify(canvas.getDrawing())[0];
                    Toast.makeText(MainActivity.this, "result: " + result.getCharacter() + " (" + result.getConfidence() * 100 + "%)", Toast.LENGTH_LONG).show();
                } else {
                    Toast.makeText(MainActivity.this, "classifier not loaded", Toast.LENGTH_LONG).show();
                }
            }
        });

        findViewById(R.id.clear).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                preview.setImageBitmap(null);
                canvas.clear();
            }
        });
    }

    @Override
    public void onModelLoaded(DrawingsClassifier classifier) {
        this.classifier = classifier;
    }

}
