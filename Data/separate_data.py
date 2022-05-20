import os
import random

input_folder = "images"

train = []
test = []

for subfolder in os.listdir(input_folder):
    internal_folder = os.path.join(input_folder, subfolder)
    internal_files = os.listdir(internal_folder)
    for i in range(len(internal_files)):
        if ".txt" in internal_files[i]:
            del internal_files[i]
    for i in range(round(len(internal_files)*.3)):
        num = random.randint(0, len(internal_files)-1)
        test.append(f'images\{subfolder}\{internal_files[num]}')
        del internal_files[num]
    for file in internal_files:
        train.append(f'images\{subfolder}\{file}')

print(test)
print(train)

print(len(test))
print(len(train))

with open("train2.txt", 'w') as train_file:
    for filename in train:
        train_file.write(filename)
        train_file.write("\n")

with open("test2.txt", 'w') as test_file:
    for filename in test:
        test_file.write(filename)
        test_file.write("\n")