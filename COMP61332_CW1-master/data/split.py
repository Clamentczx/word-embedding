import random

line_list = []

# read origin train data
with open('data/train_origin.txt') as f:
    for row in f:
        line_list.append(row)

# shuffle data 
random.shuffle(line_list)


# split data by ratio 9:1
def list_splitter(list_to_split, ratio):
    first_half = int(len(list_to_split) * ratio)
    return list_to_split[:first_half], list_to_split[first_half:]

train, dev = list_splitter(line_list, 0.9)

# store train and dev as txt file
with open('data/train.txt', 'w') as f:
    for line in train:
        f.write(f"{line}")

with open('data/dev.txt', 'w') as f:
    for line in dev:
        f.write(f"{line}")