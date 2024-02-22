import LogisticRegression.*;
import Utils.DataSet;

import java.util.Scanner;

public class Main {
    static double learningRate = 0.005;
    static int numIterations = 500;
    static double threshold = 0.5; // Threshold for classification
    static double testSize = 0.2; // Portion of the test subset

    static DataSet ds;
    static double[][] XTrain;
    static int[] YTrain;
    static double[][] XTest;
    static int[] YTest;

    static final int NUM_EVAL_RUNS = 3;
    static Scanner in;

    public static void main(String[] args) {

        if (args.length != 1) {
            System.out.println("Wrong command: there should be 1 parameter: path to the input file");
            return;
        }
        String inputFile = args[0];

        System.out.println("\nJOKER PROJECT: LOGISTIC REGRESSION\n");

        // Load dataset
        loadDataset(inputFile);

        // Using Scanner for Getting Input from User
        in = new Scanner(System.in);

        System.out.print("\nPress Enter to continue...");
        in.nextLine();

        // MENU
        char opt;

        do {
            System.out.println("\n\nChoose an option:\n");
            System.out.println("1. Train model with Sequential Logistic Regression");
            System.out.println("2. Train model with Parallel Logistic Regression");
            System.out.println("3. Evaluate Parallel Performance");
            System.out.println("4. Find best dataset size for parallelism");
            System.out.println("5. Set hyper-parameters");
            System.out.println("q. Quit");

            System.out.print("> ");
            String line = in.nextLine();
            opt = (!line.isEmpty()) ? line.charAt(0) : ' ';

            switch (opt) {
                case '1':
                    trainWithSequentialVersion();
                    break;
                case '2':
                    trainWithParallelVersion();
                    break;
                case '3':
                    evaluateParallelPerformance();
                    break;
                case '4':
                    findOptimalDatasetSizeForParallelism();
                    break;
                case '5':
                    setHyperparameters();
                    break;
                case 'q':
                case 'Q':
                    break;
                default:
                    System.out.println("\nInvalid option");
            }
            System.out.print("\nPress Enter to continue...");
            in.nextLine();
        } while (opt != 'q' && opt != 'Q');
    }

    static void loadDataset(String inputFile) {
        // Read csv file
        System.out.print("Loading dataset...");
        ds = new DataSet();
        ds.readDataSet(inputFile);
        System.out.println(" DONE");

        // Scale predictor variables
        System.out.print("Scaling data...");
        ds.normalize();
        System.out.println(" DONE");

        // Split data into training and test sets
        System.out.print("Splitting data into training and test sets...");
        ds.splitData(testSize);
        System.out.println(" DONE");

        // Print data set information
        ds.printDataSetInfo();

        // Variables
        XTrain = ds.getXTrain();
        YTrain = ds.getYTrain();
        XTest = ds.getXTest();
        YTest = ds.getYTest();
    }

    static void trainWithSequentialVersion() {
        // Create instance of the sequential logistic regression
        SequentialLogisticRegression seqLogistic = new SequentialLogisticRegression(
                XTrain[0].length, learningRate, numIterations, threshold
        );

        // Train model
        System.out.println("\nTraining model with Batch Gradient Descent");
        seqLogistic.trainModelWithBGD(XTrain, YTrain);
        System.out.println("Training DONE\n");

        // Print model weights
        seqLogistic.printModel();

        // Compute errors
        double[] predictedY = seqLogistic.scoreData(XTest);
        seqLogistic.evaluateModel(YTest, predictedY);
    }

    static void trainWithParallelVersion() {
        // Create instance of the parallel logistic regression
        ParallelLogisticRegression parLogistic = new ParallelLogisticRegression(
                XTrain[0].length, learningRate, numIterations, threshold
        );

        // Train model
        System.out.println("\nTraining model with Batch Gradient Descent");
        parLogistic.trainModelWithBGD(XTrain, YTrain);
        System.out.println("Training DONE\n");

        // Print model weights
        parLogistic.printModel();

        // Compute errors
        double[] predictedY = parLogistic.scoreData(XTest);
        parLogistic.evaluateModel(YTest, predictedY);
    }

    static void evaluateParallelPerformance() {
        System.out.println("\nEvaluating Sequential Implementation...");
        // Create instance of the sequential logistic regression
        SequentialLogisticRegression seqLogistic = new SequentialLogisticRegression(
                XTrain[0].length, learningRate, numIterations, threshold
        );

        double sequentialTime = 0;
        for (int i = 0; i < NUM_EVAL_RUNS; i++) {
            long start = System.currentTimeMillis();
            // Train model
            System.out.println("\nTraining model with Batch Gradient Descent");
            seqLogistic.trainModelWithBGD(XTrain, YTrain);
            System.out.println("Training DONE\n");
            sequentialTime += System.currentTimeMillis() - start;
        }
        sequentialTime /= NUM_EVAL_RUNS;


        System.out.println("\nEvaluating Parallel Implementation...");
        // Create instance of the sequential logistic regression
        ParallelLogisticRegression parLogistic = new ParallelLogisticRegression(
                XTrain[0].length, learningRate, numIterations, threshold
        );

        double parallelTime = 0;
        for (int i = 0; i < NUM_EVAL_RUNS; i++) {
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

        double speedup = sequentialTime / parallelTime;
        int numProcessors = Runtime.getRuntime().availableProcessors();
        double efficiency = 100 * speedup / numProcessors;

        System.out.println();
        System.out.println("Performance Evaluation: ");
        System.out.format("Average Sequential Time: %.1f ms\n", sequentialTime);
        System.out.format("Average Parallel Time: %.1f ms\n", parallelTime);
        System.out.format("Speedup: %.2f \n", speedup);
        System.out.println("Number of processors: " + numProcessors);
        System.out.format("Efficiency: %.2f%%\n", efficiency);
    }

    static void setHyperparameters() {
        System.out.println("\n\nEnter hyperparameters: ");
        System.out.print("learning rate (" + learningRate + "): ");
        learningRate = in.nextDouble();
        System.out.print("number of iterations (" + numIterations + "): ");
        numIterations = in.nextInt();
        System.out.print("threshold (" + threshold + "): ");
        threshold = in.nextDouble();
    }

    static void findOptimalDatasetSizeForParallelism() {
        int nFeatures = 10;
        int nIterations = 90;
        int minObservations = 500;
        int maxObservations = 100_000;
        int numRuns = 10;
        double speedup = 0;

        for (int n = minObservations; n <= maxObservations; n += 500) {
            System.out.println("\n\nEvaluating dataset with " + n + " observations...");
            DataSet myDs = new DataSet();
            myDs.generateDataset(n, nFeatures);
            double[][] X = myDs.getX();
            int[] Y = myDs.getY();

            // Create instance of the sequential logistic regression
            SequentialLogisticRegression seqLogistic = new SequentialLogisticRegression(
                    nFeatures, learningRate, nIterations, threshold
            );

            double sequentialTime = 0;
            for (int i = 0; i < numRuns; i++) {
                long start = System.currentTimeMillis();
                seqLogistic.trainModelWithBGD(X, Y);
                sequentialTime += System.currentTimeMillis() - start;
            }
            sequentialTime /= numRuns;

            // Create instance of the sequential logistic regression
            ParallelLogisticRegression parLogistic = new ParallelLogisticRegression(
                    nFeatures, learningRate, nIterations, threshold
            );

            double parallelTime = 0;
            for (int i = 0; i < numRuns; i++) {
                long start = System.currentTimeMillis();
                parLogistic.trainModelWithBGD(X, Y);
                parallelTime += System.currentTimeMillis() - start;
            }
            parallelTime /= numRuns;

            speedup = sequentialTime / parallelTime;
            int numProcessors = Runtime.getRuntime().availableProcessors();
            double efficiency = 100 * speedup / numProcessors;

            System.out.println();
            System.out.println("Performance Evaluation: ");
            System.out.format("Average Sequential Time: %.1f ms\n", sequentialTime);
            System.out.format("Average Parallel Time: %.1f ms\n", parallelTime);
            System.out.format("Speedup: %.2f \n", speedup);
            System.out.format("Efficiency: %.2f%%\n", efficiency);

            if (speedup > 1.0) {
                System.out.println("\nApproximate optimal dataset size for parallelism: " + n);
                break;
            }
        }
    }
}