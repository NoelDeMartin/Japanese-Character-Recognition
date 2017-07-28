package com.lincolnschilli.jcr.views;

import android.content.Context;
import android.graphics.Canvas;
import android.support.annotation.Nullable;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import com.lincolnschilli.jcr.drawing.Drawing;

public class DrawingSheetView extends View implements View.OnTouchListener {

    private Drawing drawing;

    public DrawingSheetView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        this.drawing = new Drawing(getWidth(), getHeight());
        this.setOnTouchListener(this);
    }

    public Drawing getDrawing() {
        return drawing;
    }

    public void clear() {
        drawing.clear();
        invalidate();
    }

    @Override
    protected void onSizeChanged(int width, int height, int oldWidth, int oldHeight) {
        super.onSizeChanged(width, height, oldWidth, oldHeight);
        drawing.resize(width, height);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        drawing.draw(canvas);
    }

    @Override
    public boolean onTouch(View view, MotionEvent motionEvent) {

        int action = motionEvent.getAction() & MotionEvent.ACTION_MASK;

        float x = motionEvent.getX();
        float y = motionEvent.getY();

        switch (action) {
            case MotionEvent.ACTION_DOWN:
                drawing.startStroke(x, y);
                invalidate();
                return  true;
            case MotionEvent.ACTION_MOVE:
                drawing.continueStroke(x, y);
                invalidate();
                return true;
            case MotionEvent.ACTION_UP:
                drawing.completeStroke(x, y);
                invalidate();
                return true;
            default:
                return false;
        }

    }

}
