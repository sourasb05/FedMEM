# Original dictionary
original_dict = {
    "row 0": [0.34, 0.15, 0.37, 0.34, 0.15, 0.37 ],
    "row 1": [0.15, 0.51, 0.21, 0.15, 0.51, 0.21 ],
    "row 2": [0.23, 0.41, 0.10, 0.23, 0.41, 0.10]
}

# Step 1: Find the minimum in each column
min_in_columns = [min(original_dict[row][i] for row in original_dict) for i in range(len(original_dict["row 0"]))]

# Step 2: Create new dictionary with 1s and 0s
result_dict = {}
for row in original_dict:
    result_dict[row] = [1 if original_dict[row][i] == min_in_columns[i] else 0 for i in range(len(original_dict[row]))]

print(original_dict)
print(min_in_columns)
print(result_dict)


# List containing the column IDs
column_id = [3, 2, 0, 5, 4, 1]

# Create a new dictionary with column IDs
column_id_dict = {}
for row_key, row_values in result_dict.items():
    column_id_dict[row_key] = [column_id[i] for i, value in enumerate(row_values) if value == 1]

print(column_id_dict)

"""# Step 3: Create a new dictionary with column numbers
column_dict = {}
for row_key, row_values in result_dict.items():
    column_dict[row_key] = [i for i, value in enumerate(row_values) if value == 1]

# Step 4: Create a list of lists from the new dictionary
list_of_lists = list(column_dict.values())

print(column_dict)
print(list_of_lists)
"""