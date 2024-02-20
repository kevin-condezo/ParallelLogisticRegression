package LogisticRegression;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Arrays;
import java.util.Locale;

public class SequentialLogisticRegression {

    protected double[] weights;
    protected int numFeatures;
    protected double learningRate;
    protected int numIterations;
    protected double threshold; // Threshold to determine the class (between 0 and 1)

    public SequentialLogisticRegression(int numFeatures, double learningRate, int numIterations, double threshold) {
        this.numFeatures = numFeatures;
        this.learningRate = learningRate;
        this.numIterations = numIterations;
        this.threshold = threshold;
    }

    /**
     * Sigmoid function
     */
    protected double sigmoid(double z) {
        return (1.0 / (1.0 + Math.exp(-z)));
    }

    /**
     * Training using Batch Gradient Descent
     */
    public void trainModelWithBGD(double[][] X, int[] Y) {
        weights = new double[numFeatures]; // filled with zeros
        // Iterate until maxIterations or the STOP condition is met
        for (int n = 0; n < numIterations; n++) {
            double[] gradient = new double[numFeatures];

            // Compute gradient for each feature
            for (int i = 0; i < X.length; i++) {
                double yPredicted = computePrediction(X[i]);
                double error = yPredicted - Y[i];

                for (int j = 0; j < numFeatures; j++) {
                    gradient[j] += error * X[i][j];
                }
            }

            // Update weights using the gradient
            for (int j = 0; j < numFeatures; j++) {
                weights[j] -= learningRate * gradient[j] / X.length;
            }

            if ((n+1) % 100 == 0)
                System.out.println("Iteration " + (n+1) + ": gradient = " + Arrays.toString(gradient));
        }
    }

    private double computePrediction(double[] x) {
        double z = 0.0;
        for (int i = 0; i < numFeatures; i++) {
            z += weights[i] * x[i];
        }
        return sigmoid(z);
    }

    public double[] scoreData(double[][] data) {
        int n = data.length;
        double[] predictedY = new double[n];
        for (int i = 0; i < n; i++) {
            predictedY[i] = computePrediction(data[i]);
        }
        return predictedY;
    }

    public void evaluateModel(int[] Y, double[] predictedY) {
        int FP = 0;
        int FN = 0;
        int TP = 0;
        int TN = 0;
        double precision;
        double recall;
        double accuracy;

        for (int i = 0; i < predictedY.length; i++) {
            int predY = ((predictedY[i] >= threshold) ? 1 : 0);

            if ((Y[i] == 1) && (predY == 1)) TP++;
            else if ((Y[i] == 0) && (predY == 1)) FP++;
            else if ((Y[i] == 0) && (predY == 0)) TN++;
            else if ((Y[i] == 1) && (predY == 0)) FN++;
        }

        precision = 1.0 * TP / (TP + FP);
        recall = 1.0 * TP / (TP + FN);
        accuracy = 1.0 * (TP + TN) / (TP + TN + FP + FN);

        DecimalFormat df = new DecimalFormat("##.###");
        df.setDecimalFormatSymbols(new DecimalFormatSymbols(Locale.US));

        System.out.println();
        System.out.println("Performance report:");
        System.out.println("TP= " + TP + " FP= " + FP + " TN= " + TN + " FN= " + FN);
        System.out.println("Precision= " + df.format(precision));
        System.out.println("Recall= " + df.format(recall));
        System.out.println("Accuracy= " + df.format(accuracy));
        System.out.println();
    }

    public void printModel() {
        DecimalFormat df = new DecimalFormat("###.####");
        df.setDecimalFormatSymbols(new DecimalFormatSymbols(Locale.US));
        System.out.println("Sequential logistic regression weights:");
        for (double weight : weights) {
            System.out.print(df.format(weight) + " ");
        }
        System.out.println();
    }

    public double[] getWeights() {
        return weights;
    }
}