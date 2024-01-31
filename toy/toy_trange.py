from tqdm import trange

# Define the number of experiments (assuming it's 100)
num_experiments = 100

# Use trange and manually format the description
for i in trange(num_experiments, desc="Experiment Progress"):
    # Calculate the progress percentage
    progress_percentage = (i + 1) * 100 / num_experiments
    
    # Print the experiment number and progress percentage
    print(f"Experiment {i + 1} {progress_percentage:.1f}%")
    
    # Your code for each experiment goes here