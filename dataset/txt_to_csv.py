import csv
def create_csv_file(input_file, output_file):
    """Creates a CSV file from a text file that contains two columns separated by space.

    Args:
        input_file: The path to the input text file.
        output_file: The path to the output CSV file.
    """

    with open(input_file, "r") as f_in, open(output_file, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        for line in f_in:
            values = line.split()
            writer.writerow(values)

if __name__ == "__main__":
    input_file = "/proj/sourasb-220503/FedMEM/dataset/tensor_train_val_1.txt"
    output_file = "/proj/sourasb-220503/FedMEM/dataset/tensor_train_val_1.csv"

    create_csv_file(input_file, output_file)