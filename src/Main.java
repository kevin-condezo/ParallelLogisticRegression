import LogisticRegression.*;
import Utils.DataSet;

public class Main {
    public static void main(String[] args) {

        if (args.length != 1) {
            System.out.println("Wrong command: there should be 1 parameter: path to the input file");
            return;
        }
        String inputFile = args[0];

        // Read csv file
        System.out.print("Loading data...");
        DataSet inputData = new DataSet();
        inputData.readDataSet(inputFile);
        System.out.println(" DONE");

        // Hyper parameters
        double learningRate = 0.005;
        int numIterations = 500;
        double threshold = 0.5; // Threshold for classification
        double testSize = 0.2; // Portion of the test set

        // Scale predictor variables
        System.out.print("Scaling data...");
        inputData.normalize();
        System.out.println(" DONE");

        // Split data into training and test sets
        System.out.print("Splitting data into training and test sets...");
        inputData.splitData(testSize);
        System.out.println(" DONE");

        // Variables
        double[][] XTrain = inputData.getXTrain();
        int[] YTrain = inputData.getYTrain();
        double[][] XTest = inputData.getXTest();
        int[] YTest = inputData.getYTest();


        final int NUM_EVAL_RUNS = 3;

        System.out.println("Evaluating Sequential Implementation...");
        // Create instance of the sequential logistic regression
        SequentialLogisticRegression seqLogistic = new SequentialLogisticRegression(
                XTrain[0].length, learningRate, numIterations, threshold
        );

        double sequentialTime = 0;
        for(int i=0; i<NUM_EVAL_RUNS; i++) {
            long start = System.currentTimeMillis();
            // Train model
            System.out.println("\nTraining model with Batch Gradient Descent");
            seqLogistic.trainModelWithBGD(XTrain, YTrain);
            System.out.println("Training DONE\n");
            sequentialTime += System.currentTimeMillis() - start;
        }
        sequentialTime /= NUM_EVAL_RUNS;

        System.out.println("Evaluating Parallel Implementation...");
        // Create instance of the sequential logistic regression
        ParallelLogisticRegression parLogistic = new ParallelLogisticRegression(
                XTrain[0].length, learningRate, numIterations, threshold
        );

        double parallelTime = 0;
        for(int i=0; i<NUM_EVAL_RUNS; i++) {
            long start = System.currentTimeMillis();
            // Train model
            System.out.println("\nTraining model with Batch Gradient Descent");
            parLogistic.trainModelWithBGD(XTrain, YTrain);
            System.out.println("Training DONE\n");
            parallelTime += System.currentTimeMillis() - start;
        }
        parallelTime /= NUM_EVAL_RUNS;

        // Print models weights
        seqLogistic.printModel();
        parLogistic.printModel();

        // Compute errors
        double[] predictedY = parLogistic.scoreData(XTest);
        parLogistic.evaluateModel(YTest, predictedY);

        System.out.println();
        System.out.format("Average Sequential Time: %.1f ms\n", sequentialTime);
        System.out.format("Average Parallel Time: %.1f ms\n", parallelTime);
        System.out.format("Speedup: %.2f \n", sequentialTime/parallelTime);
        System.out.format("Efficiency: %.2f%%\n", 100*(sequentialTime/parallelTime)/Runtime.getRuntime().availableProcessors());
    }
}