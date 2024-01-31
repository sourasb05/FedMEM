import os

def read_jpg_strings_from_file(file_path):
    train_path = file_path + "train.txt"
    valid_path = file_path + "validation.txt"
    r3_path = file_path + "r3/"
    train_images_file = []
    train_file_lines = []
    with open(train_path, "r") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            if len(words) >0:
                train_images_file.append(words[0])
                train_file_lines.append(line)
        print(len(train_images_file))
    # print(train_file_lines)
    # input("press")
    file_names = os.listdir(r3_path)
    jpg_file_names = []
    for file_name in file_names:
        if file_name.endswith(".jpg"):
            jpg_file_names.append(file_name)
    #print(len(jpg_file_names))

    set_A = set(train_images_file)
    set_B = set(jpg_file_names)

    if set_A.issubset(set_B) is True:
        print("1")
    else:
        print("0")

    common_elements = set_A.intersection(set_B)
    # print(list(common_elements))

    list_c = []
    for element_a in train_file_lines:
        for element_b in common_elements:
            if element_b in element_a:
                list_c.append(element_a)
    print(len(list_c))

    text_file_path = file_path + "refined_training_file.txt"
    with open(text_file_path, "w") as f:
        for element in list_c:
            f.write(element + "\n")

if __name__ == "__main__":
  
  file_path = "/proj/sourasb-220503/Predicting-Event-Memorability/FedMEM/dataset/"
  read_jpg_strings_from_file(file_path)
