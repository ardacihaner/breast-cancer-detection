import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-0.005 * x))


def sigmoid_derivative(x):
    return 0.005 * x * (1 - x)


def read_and_divide_into_train_and_test(csv_file):
    # Reading csv file here
    df = pd.read_csv(csv_file)
    # Dropping unnecessary column
    df.drop(['Code_number'], axis=1, inplace=True)
    # Replacing missing values in the Bare Nuclei column with mean of rest of the values
    df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei'], errors='coerce')
    mean_missing = int(round(df['Bare_Nuclei'].mean()))
    df['Bare_Nuclei'] = df['Bare_Nuclei'].replace(np.NaN, mean_missing).astype(int)

    # Splitting dataframe into testing and training dataframes
    training_df = df.sample(frac=0.8, random_state=0)
    test_df = df.drop(training_df.index)
    training_inputs = training_df.iloc[:, :-1]
    training_labels = training_df.iloc[:, -1]

    test_inputs = test_df.iloc[:, :-1]
    test_labels = test_df.iloc[:, -1]

    # Creating the correlation heatmap of the dataframe
    df.drop(['Class'], axis=1, inplace=True)
    correlation = df.corr()
    plt.figure(figsize=(10, 10))
    heatmap = plt.imshow(correlation, cmap='hot')
    plt.xticks(range(len(correlation)), correlation, rotation=90)
    plt.yticks(range(len(correlation)), correlation)
    for i in range(len(correlation)):
        for j in range(len(correlation)):
            if round(correlation.iloc[i, j], 2) > .5:
                color = 'k'
            else:
                color = 'w'
            plt.text(j, i, round(correlation.iloc[i, j], 2),
                     ha="center", va="center", color=color)

    plt.fill()
    plt.colorbar(heatmap)
    print("Please close the heatmap to continue...")
    plt.show()
    plt.close()

    return training_inputs, training_labels, test_inputs, test_labels


def run_on_test_set(test_inputs, test_labels, weights):
    test_output = sigmoid(test_inputs.dot(weights))
    tp = 0
    test_predictions = []
    for i, j in test_output.iterrows():
        j = float(j)
        if j < 0.5:
            test_predictions.append(0)
        elif j >= 0.5:
            test_predictions.append(1)
    for predicted_val, label in zip(test_predictions, test_labels):
        if predicted_val == label:
            tp += 1
    # accuracy = tp_count / total number of samples
    accuracy = tp / len(test_labels)
    return accuracy


def plot_loss_accuracy(accuracy_array, loss_array):
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(accuracy_array)
    ax1.set_title('Accuracy')

    ax2.plot(loss_array)
    ax2.set_title('Loss')
    plt.show()



def main():
    csv_file = './breast-cancer-wisconsin.csv'
    pd.read_csv(csv_file)
    iteration_count = 2500
    np.random.seed(1)
    weights = 2 * np.random.random((9, 1)) - 1
    accuracy_array = []
    loss_array = []
    training_inputs, training_labels, test_inputs, test_labels = read_and_divide_into_train_and_test(csv_file)
    print("Calculating... ")
    training_inputs = np.array(training_inputs)
    training_outputs = np.array([training_labels])
    training_outputs = training_outputs.transpose()
    for iteration in range(iteration_count):
        outputs = training_inputs.dot(weights)
        outputs = sigmoid(outputs)
        loss = np.subtract(training_outputs, outputs)
        tunings = loss * sigmoid_derivative(outputs)
        weights = weights + training_inputs.transpose().dot(tunings)
        accuracy = run_on_test_set(test_inputs, test_labels, weights)
        accuracy_array.append(accuracy)
        loss_array.append(np.mean(loss))
    plot_loss_accuracy(accuracy_array, loss_array)


if __name__ == '__main__':
    main()
