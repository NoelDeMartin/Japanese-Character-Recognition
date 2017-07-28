package com.lincolnschilli.jcr.drawing;

import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;

public class Stroke {

    private Path path;

    public Stroke(float startX, float startY) {
        this.path = new Path();
        path.moveTo(startX, startY);
    }

    public void transform(Matrix matrix) {
        path.transform(matrix);
    }

    public void addPoint(float x, float y) {
        path.lineTo(x, y);
    }

    public void draw(Canvas canvas) {
        draw(canvas, Tools.defaultBrush);
    }

    public void draw(Canvas canvas, Paint paint) {
        canvas.drawPath(path, paint);
    }

}
