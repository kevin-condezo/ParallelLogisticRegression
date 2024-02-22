package Utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.util.Arrays;

public class DataSet {
    // Full dataset
    private double[][] X; // features
    private int[] Y; // label

    // Training and test subsets
    private double[][] XTrain;
    private int[] YTrain;
    private double[][] XTest;
    private int[] YTest;

    private static final String DELIMITER = ","; // used in CSV files
    protected String[] varNames;

    private int countLines(String filename) throws IOException {
        try (InputStream is = new BufferedInputStream(new FileInputStream(filename))) {
            byte[] c = new byte[1024];
            int count = 0;
            int readChars;
            boolean empty = true;
            while ((readChars = is.read(c)) != -1) {
                empty = false;
                for (int i = 0; i < readChars; ++i) {
                    if (c[i] == '\n') {
                        count++;
                    }
                }
            }
            return (count == 0 && !empty) ? 1 : count;
        }
    }

    /**
     * Read the dataset
     * Given that:
     * 1) it contains a header
     * 2) first column contains the row index
     * 3) second column contains the target
     * 4) columns are separated with ","
     */
    public void readDataSet(String fileName) {
        BufferedReader fileReader = null;

        try {
            // First, determine the number of lines in the file
            int linesNumber = countLines(fileName);

            // allocate memory for rows
            X = new double[linesNumber - 1][];
            Y = new int[linesNumber - 1];

            String line;
            // Create the file reader
            fileReader = new BufferedReader(new FileReader(fileName));

            // Read the variable names
            line = fileReader.readLine();
            varNames = line.split(DELIMITER);

            // Read the file line by line
            int rowNumber = 0;
            while ((line = fileReader.readLine()) != null) {
                // Get all columns available in line
                String[] columns = line.split(DELIMITER);

                double[] x = new double[columns.length - 1];

                // Add bias as x[0]
                x[0] = 1.0;

                // Skip first column (it is the observation number)
                // also: start from x[1]
                for (int i = 2; i < columns.length; i++) {
                    x[i - 1] = Double.parseDouble(columns[i]);
                }

                // store values in memory
                X[rowNumber] = x;
                Y[rowNumber++] = Integer.parseInt(columns[1]);

            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                assert fileReader != null;
                fileReader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * Normalize the data
     */
    public void normalize() {
        double[] varX = new double[X.length];
        for (int i = 0; i < X[0].length; i++) {
            for (int j = 0; j < X.length; j++) {
                varX[j] = X[j][i];
            }
            double maxVal = Arrays.stream(varX).max().getAsDouble();
            double minVal = Arrays.stream(varX).min().getAsDouble();
            if (maxVal == minVal)
                continue;
            for (int j = 0; j < X.length; j++) {
                X[j][i] = (X[j][i] - minVal) / (maxVal - minVal);
            }
        }
    }

    /**
     * Split the data into training and test sets
     */
    public void splitData(double testSize) {
        int n = X.length;
        int testSizeInt = (int) (n * testSize);
        XTest = Arrays.copyOfRange(X, 0, testSizeInt);
        YTest = Arrays.copyOfRange(Y, 0, testSizeInt);
        XTrain = Arrays.copyOfRange(X, testSizeInt, n);
        YTrain = Arrays.copyOfRange(Y, testSizeInt, n);
    }

    public void printDataSetInfo() {
        System.out.println("\nData set information:");
        System.out.println("Number of observations: " + X.length);
        System.out.println("Number of features: " + X[0].length);
        System.out.println("Number of training observations: " + XTrain.length);
        System.out.println("Number of test observations: " + XTest.length);
        System.out.println();
        System.out.println("First 5 rows of the dataset: ");
        for (int i = 0; i < 5; i++) {
            System.out.println(Arrays.toString(X[i]) + "\t\t" + Y[i]);
        }
    }

    public void generateDataset(int nObservations, int nFeatures) {
        X = new double[nObservations][nFeatures];
        Y = new int[nObservations];
        for (int i = 0; i < nObservations; i++) {
            X[i][0] = 1.0;
            for (int j = 1; j < nFeatures; j++) {
                X[i][j] = Math.random();
            }
            Y[i] = Math.random() > 0.5 ? 1 : 0;
        }
    }

    public double[][] getX() {
        return X;
    }

    public int[] getY() {
        return Y;
    }

    public double[][] getXTrain() {
        return XTrain;
    }

    public int[] getYTrain() {
        return YTrain;
    }

    public double[][] getXTest() {
        return XTest;
    }

    public int[] getYTest() {
        return YTest;
    }
}