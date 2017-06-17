package com.lincolnschilli.jcr.drawing;

import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;

public class Stroke {

    private Path path;
    private static Paint defaultPaint = new Paint();

    static {
        defaultPaint.setStyle(Paint.Style.STROKE);
        defaultPaint.setStrokeWidth(8);
    }

    public Stroke(float startX, float startY) {
        this.path = new Path();
        path.moveTo(startX, startY);
    }

    public void addPoint(float x, float y) {
        path.lineTo(x, y);
    }

    public void draw(Canvas canvas) {
        draw(canvas, defaultPaint);
    }

    public void draw(Canvas canvas, Paint paint) {
        canvas.drawPath(path, paint);
    }

}
