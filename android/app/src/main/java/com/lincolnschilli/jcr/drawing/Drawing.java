package com.lincolnschilli.jcr.drawing;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;

import java.util.ArrayList;

public class Drawing {

    private ArrayList<Stroke> strokes;
    private Stroke currentStroke;
    private float width, height;

    private Matrix transformation;

    public Drawing(float width, float height) {
        this.strokes = new ArrayList<>();
        this.width = width;
        this.height = height;
        transformation = new Matrix();
    }

    public void resize(int width, int height) {
        if (this.width > 0 && this.height > 0) {
            scaleStrokes(width / this.width, height / this.height);
        }
        this.width = width;
        this.height = height;
    }

    public void clear() {
        strokes.clear();
    }

    public void draw(Canvas canvas) {
        draw(canvas, Tools.defaultBrush);
    }

    public void draw(Canvas canvas, Paint paint) {
        for (Stroke stroke: strokes) {
            stroke.draw(canvas, paint);
        }
    }

    public void drawBinaryData(float[] buffer, int width, int height) {

        Bitmap bitmap = this.getBitmap(width, height);

        // Write bitmap data to buffer
        int dataLength = width*height;
        int[] pixels = new int[dataLength];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        for (int i = 0; i < dataLength; i++) {
            buffer[i] = pixels[i] == 0? 0 : 1;
        }

    }

    public Bitmap getBitmap(int width, int height) {
        // Transform strokes to buffer dimensions
        scaleStrokes(width / this.width, height / this.height);

        // Draw in bitmap
        Paint brush = new Paint();
        brush.setStyle(Paint.Style.STROKE);
        brush.setStrokeWidth(3);
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ALPHA_8);
        Canvas canvas = new Canvas(bitmap);
        draw(canvas, brush);

        // Restore stroke dimensions
        scaleStrokes(this.width / width, this.height / height);

        return bitmap;
    }

    public void startStroke(float x, float y) {
        currentStroke = new Stroke(x, y);
        strokes.add(currentStroke);
    }

    public void continueStroke(float x, float y) {
        currentStroke.addPoint(x, y);
    }

    public void completeStroke(float x, float y) {
        currentStroke.addPoint(x, y);
        currentStroke = null;
    }

    private void scaleStrokes(float scaleX, float scaleY) {
        transformation.setScale(scaleX, scaleY);
        for (Stroke stroke: strokes) {
            stroke.transform(transformation);
        }
    }

}
