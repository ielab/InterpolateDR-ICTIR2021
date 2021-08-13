import os
import random
import tqdm
base_dir = os.getcwd() + '/'
file_dir = base_dir + "Msmarco/ance/"  # input folder, modify this if want to test on other dataset
feature_file = file_dir + "feature_out/features.txt"
prob_set = 0.90

output_train_bce = open(file_dir + "feature_out/train_bce.txt", 'w')
output_train_bound = open(file_dir + "feature_out/train_bound.txt", 'w')
output_train_ave = open(file_dir + "feature_out/train_ave.txt", 'w')
output_test = open(file_dir + "feature_out/test.txt", 'w')

feature_input = open(feature_file, 'r').readlines()

for line in feature_input:
    current_num = random.random()
    if current_num>prob_set:
        output_test.write(line)
    else:
        output_train_bce.write(line)
        items = line.split(sep = '\t')
        y = items[20:31]
        y_set = []
        for i,y_value in enumerate(y):
            if int(y_value)==1:
                y_set.append(i/10)
        for item in items[0:20]:
            output_train_bound.write(item + '\t')
            output_train_ave.write(item+'\t')
        output_train_bound.write(str(min(y_set)) + '\t')
        output_train_bound.write(str(max(y_set))+ '\t\n')
        output_train_ave.write(str((max(y_set)+min(y_set))/2) + '\t\n')









