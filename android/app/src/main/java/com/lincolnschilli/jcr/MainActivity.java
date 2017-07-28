package com.lincolnschilli.jcr;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Toast;

import com.lincolnschilli.jcr.classification.DrawingsClassifier;
import com.lincolnschilli.jcr.views.DrawingSheetView;

public class MainActivity extends AppCompatActivity {

    private DrawingsClassifier classifier;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {
            this.classifier = new DrawingsClassifier(this, "categories.csv", "model_katakana8_90.pb");
        } catch (Exception e) {
            Toast.makeText(this, "There was an error loading model files!", Toast.LENGTH_LONG);
            e.printStackTrace();
        }

        findViewById(R.id.classify).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                classifier.classify(((DrawingSheetView) findViewById(R.id.canvas)).getDrawing());
            }
        });

        findViewById(R.id.clear).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                ((DrawingSheetView) findViewById(R.id.canvas)).clear();
            }
        });
    }

}
