package com.lincolnschilli.jcr.drawing;

import android.graphics.Paint;

public class Tools {

    public static Paint defaultBrush = new Paint();

    static {
        defaultBrush.setStyle(Paint.Style.STROKE);
        defaultBrush.setStrokeWidth(8);
    }

}
