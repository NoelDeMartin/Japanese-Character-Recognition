package com.lincolnschilli.jcr.classification.tasks;

import com.lincolnschilli.jcr.classification.DrawingsClassifier;

public interface LoadModelTaskListener {

    void onModelLoaded(DrawingsClassifier classifier);

}
