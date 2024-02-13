import LogisticRegression.*;
import Utils.DataSet;

public class Main {
    public static void main(String[] args) {

        if (args.length != 1) {
            System.out.println("Wrong command: there should be 1 parameters: path to the input file");
            return;
        }

        // Input file which needs to be parsed
        String inputFile = args[0];

        // Output file: input file with the appended predictions
        String outputFile = args[0] + ".out";

        // Optimization parameters
        double learningRate = 0.0001;    // Learning rate
        int maxIterations = 100000;        // Maximum number of iterations
        double minDelta = 0.0001;        // Minimum change of the weights (STOP criteria)
        double cutOff = 0.5;            // Classification cut off

        // Read input data from csv file
        System.out.print("Loading data...");
        DataSet inputData = new DataSet();
        inputData.readDataSet(inputFile);
        System.out.println(" DONE.");

        // Print input data
        //inputData.printDataSet();

        // Predictor variables
        double[][] X = inputData.getX();

        // Predicted variable
        int[] Y = inputData.getY();

        // Create instance of the logistic regression
        SequentialLogisticRegression logistic = new SequentialLogisticRegression(X[0].length, learningRate, maxIterations, minDelta, cutOff);

        // Standardize predictor variables
        System.out.print("Scaling data...");
        logistic.standardize(X);
        System.out.println(" DONE.");

        // Train model
        System.out.println("Training model with Stochastic Gradient Descent");
        logistic.trainModelWithSGD(X, Y);

        System.out.println("Training DONE.");
        System.out.println();

        // Print model
        logistic.printModel();

        // Print model
        double[] predictedY = logistic.scoreData(X);

        // Compute errors
        logistic.computeErrors(Y, predictedY);

        // Save data with the appended predictions
        inputData.writeDataSetPred(outputFile, predictedY);

    }
}