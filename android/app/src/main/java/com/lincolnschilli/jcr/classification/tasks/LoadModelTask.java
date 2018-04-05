package com.lincolnschilli.jcr.classification.tasks;

import android.app.ProgressDialog;
import android.content.Context;
import android.os.AsyncTask;
import android.widget.Toast;

import com.lincolnschilli.jcr.classification.DrawingsClassifier;

public class LoadModelTask extends AsyncTask<String, Void, DrawingsClassifier> {

    private Context context;
    private LoadModelTaskListener listener;
    private ProgressDialog dialog;

    public LoadModelTask(Context context, LoadModelTaskListener listener) {
        this.context = context;
        this.listener = listener;
        this.dialog = new ProgressDialog(context);
    }

    @Override
    protected DrawingsClassifier doInBackground(String... files) {
        try {
            return new DrawingsClassifier(context, files[0], files[1]);
        } catch (Exception e) {
            Toast.makeText(context, "There was an error loading model files!", Toast.LENGTH_LONG).show();
            e.printStackTrace();
            return null;
        }
    }

    @Override
    protected void onPreExecute() {
        dialog.setMessage("Loading TensorFlow model...");
        dialog.show();
    }

    @Override
    protected void onPostExecute(DrawingsClassifier classifier) {
        if (dialog.isShowing()) {
            dialog.dismiss();
        }
        if (listener != null) {
            listener.onModelLoaded(classifier);
        }
    }

}
