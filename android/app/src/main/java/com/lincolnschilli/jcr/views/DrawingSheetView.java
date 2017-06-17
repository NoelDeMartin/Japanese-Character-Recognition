package com.lincolnschilli.jcr.views;

import android.content.Context;
import android.graphics.Canvas;
import android.support.annotation.Nullable;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import com.lincolnschilli.jcr.drawing.Stroke;

import java.util.ArrayList;

public class DrawingSheetView extends View implements View.OnTouchListener {

    private ArrayList<Stroke> strokes;
    private Stroke currentStroke;

    public DrawingSheetView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        this.strokes = new ArrayList<>();
        this.setOnTouchListener(this);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        for (Stroke stroke: strokes) {
            stroke.draw(canvas);
        }

    }

    @Override
    public boolean onTouch(View view, MotionEvent motionEvent) {

        int action = motionEvent.getAction() & MotionEvent.ACTION_MASK;

        float x = motionEvent.getX();
        float y = motionEvent.getY();

        switch (action) {
            case MotionEvent.ACTION_DOWN:
                startStroke(x, y);
                return  true;
            case MotionEvent.ACTION_MOVE:
                continuteStroke(x, y);
                return true;
            case MotionEvent.ACTION_UP:
                completeStroke(x, y);
                return true;
            default:
                return false;
        }

    }

    private void startStroke(float x, float y) {
        currentStroke = new Stroke(x, y);
        this.strokes.add(currentStroke);
        invalidate();
    }

    private void continuteStroke(float x, float y) {
        currentStroke.addPoint(x, y);
        invalidate();
    }

    private void completeStroke(float x, float y) {
        currentStroke.addPoint(x, y);
        currentStroke = null;
        invalidate();
    }

}
