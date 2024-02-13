package LogisticRegression;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;

public class SequentialLogisticRegression {
    /**
     * Weights to learn
     */
    protected double[] weights;

    /**
     * Learning rate
     */
    protected double learningRate;

    /**
     * Maximum number of iterations
     */
    protected int maxIterations;

    /**
     * Minimum change of the weights (STOP criteria)
     */
    protected double minDelta;

    /**
     * Classification cut off
     */
    protected double cutOff;

    // TODO: Revisar
//    public LogisticRegression.ParallelLogisticRegression(int n) {
//        weights = new double[n];
//        this.learningRate = 0.0001;
//        this.maxIterations = 3000;
//        this.minDelta = 0.0001;
//        this.cutOff = 0.5;
//    }

    /**
     * Constructor
     */
    public SequentialLogisticRegression(int n, double learningRate, int maxIterations, double minDelta, double cutOff) {
        weights = new double[n];
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.minDelta = minDelta;
        this.cutOff = cutOff;
    }

    /**
     * Sigmoid function
     */
    protected double sigmoid(double z) {
        return (1.0 / (1.0 + Math.exp(-z)));
    }

    /**
     * Training function
     * Use Stochastic Gradient Descent
     */
    public void trainModelWithSGD(double[][] X, int[] Y) {
        double likelihood = 0.0;
        double[] weightsPrev = new double[weights.length];
        double maxWeightDev;

        // Iterate until maxIterations or the STOP condition is met
        for (int n = 0; n < maxIterations; n++) {
            likelihood = 0.0;

            // Store previous weights
            System.arraycopy(weights, 0, weightsPrev, 0, weights.length);

            // Update weights
            for (int i = 0; i < X.length; i++) {
                double yPredicted = computePrediction(X[i]);

                for (int j = 0; j < weights.length; j++) {
                    weights[j] = weights[j] + learningRate * (Y[i] - yPredicted) * X[i][j];
                }
                // Compute log-likelihood
                likelihood += Y[i] * Math.log(computePrediction(X[i])) + (1 - Y[i]) * Math.log(1 - computePrediction(X[i]));

            }
            if (n % 5000 == 0)
                System.out.println("Iteration " + n + ": log likelihood = " + likelihood);

            // Check STOP criteria
            maxWeightDev = 0.0;
            for (int j = 0; j < weights.length; j++) {
                if ((Math.abs(weights[j] - weightsPrev[j]) / (Math.abs(weightsPrev[j]) + 0.01 * minDelta)) > maxWeightDev) {
                    maxWeightDev = (Math.abs(weights[j] - weightsPrev[j]) / (Math.abs(weightsPrev[j]) + 0.01 * minDelta));
                }
            }
            if (maxWeightDev < minDelta) {
                System.out.println("STOP criteria met: Iteration " + n + ", Log-likelihood = " + likelihood);
                break;
            }
        }
        System.out.println("Final log-likelihood= " + likelihood);
        System.out.println();
    }

    /**
     * Use the model to compute prediction for the given x
     */
    private double computePrediction(double[] x) {
        double logit = 0.0;
        for (int i = 0; i < weights.length; i++) {
            logit += weights[i] * x[i];
        }
        return sigmoid(logit);
    }

    /**
     * Print the model
     */
    public void printModel() {
        DecimalFormat df = new DecimalFormat("###.###");
        df.setDecimalFormatSymbols(new DecimalFormatSymbols(Locale.US));
        System.out.println("Logistic regression weights:");
        for (double weight : weights) {
            System.out.print(df.format(weight) + " ");
        }
        System.out.println();
    }

    /**
     * Score data with the model
     */
    public double[] scoreData(double[][] data) {
        int n = data.length;
        double[] predictedY = new double[n];

        for (int i = 0; i < n; i++) {
            predictedY[i] = computePrediction(data[i]);
        }
        return predictedY;
    }

    /**
     * Compute error rates
     */
    public void computeErrors(int[] Y, double[] predictedY) {
        int FP = 0;
        int FN = 0;
        int TP = 0;
        int TN = 0;
        double FNR;
        double FPR;

        for (int i = 0; i < predictedY.length; i++) {
            int predY = ((predictedY[i] >= cutOff) ? 1 : 0);
            if ((Y[i] == 1) && (predY == 1)) TP += 1;
            else if ((Y[i] == 0) && (predY == 1)) FP += 1;
            else if ((Y[i] == 0) && (predY == 0)) TN += 1;
            else if ((Y[i] == 1) && (predY == 0)) FN += 1;
        }

        FNR = 1.0 * FN / (TP + FN);
        FPR = 1.0 * FP / (TN + FP);

        DecimalFormat df = new DecimalFormat("##.###");
        df.setDecimalFormatSymbols(new DecimalFormatSymbols(Locale.US));

        System.out.println();
        System.out.println("Error rates (cutOff=" + cutOff + "):");
        System.out.println("False Negative Rate= " + df.format(FNR));
        System.out.println("False Positive Rate= " + df.format(FPR));
        System.out.println("TP= " + TP + " FP= " + FP + " TN= " + TN + " FN= " + FN);
        System.out.println();

    }

    private double arrayMax(double[] dataArray) {
        double maxVal = 0.0;
        for (double v : dataArray) {
            if (v > maxVal) maxVal = v;
        }
        return maxVal;
    }

    private double arrayMin(double[] dataArray) {
        double minVal = Double.MAX_VALUE;
        for (double v : dataArray) {
            if (v < minVal) minVal = v;
        }
        return minVal;
    }

    /**
     * Simple standardization
     */
    public void standardize(double[][] dataArray) {
        double[] varX = new double[dataArray.length];
        for (int i = 0; i < dataArray[0].length; i++) {
            for (int j = 0; j < dataArray.length; j++) {
                varX[j] = dataArray[j][i];
            }
            double maxVal = arrayMax(varX);
            double minVal = arrayMin(varX);
            if (maxVal == minVal)
                continue;
            for (int j = 0; j < dataArray.length; j++) {
                dataArray[j][i] = (dataArray[j][i] - minVal) / (maxVal - minVal);
            }
        }
    }
}