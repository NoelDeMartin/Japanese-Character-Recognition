package com.lincolnschilli.jcr.classification;

public class Result {

    private String character;
    private float confidence;

    public Result(String character, float confidence) {
        this.character = character;
        this.confidence = confidence;
    }

    public String getCharacter() {
        return character;
    }

    public float getConfidence() {
        return confidence;
    }

}
