# Path to the original file
original_file_path = '/proj/sourasb-220503/FedMEM/dataset/refined_training_file.txt'

# Path to the new file with modified filenames
new_file_path = '/proj/sourasb-220503/FedMEM/dataset/tensor_training_file.txt'

# Open the original file to read and the new file to write
with open(original_file_path, 'r') as original_file, open(new_file_path, 'w') as new_file:
    # Iterate through each line in the original file
    for line in original_file:
        # Split the line into filename and label
        parts = line.strip().split(' ')
        if len(parts) == 2:
            filename, label = parts
            # Replace the .jpg extension with .pt
            new_filename = filename.replace('.jpg', '.pt')
            # Write the modified filename and label to the new file
            new_file.write(f'{new_filename} {label}\n')

print(f'Modified filenames have been saved to {new_file_path}.')
