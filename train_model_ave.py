import random
import numpy as np
import time
import torch as T
import os.path
from tqdm import tqdm
#device = T.device("cpu")  # apply to Tensor or Module


# -----------------------------------------------------------

class InputDataset(T.utils.data.Dataset):
    def __init__(self, src_file,n_rows=None):

        all_xy= np.loadtxt(src_file,max_rows=n_rows,
                        usecols=[ 1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,19, 20], delimiter="\t",
                     dtype=np.float32)
        n = len(all_xy)
        tmp_x = all_xy[0:n, 0:19]  # all rows, cols [0,18)
        tmp_y = all_xy[0:n, 19:20]
        self.x_data = \
            T.tensor(tmp_x, dtype=T.float32)#.to(device)
        self.y_data = \
            T.tensor(tmp_y, dtype=T.float32)#.to(device)
        #self.truedata = \
           # T.tensor(true_d, dtype=T.int64)
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        preds = self.x_data[idx]
        trgts = self.y_data[idx]
        #true = self.truedata[idx]
        sample = {
            'predictors': preds,
            'targets': trgts
        }
        return sample
# -----------------------------------------------------------

class Net(T.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hid1 = T.nn.Linear(19, 38)  # 18-(15-15)-11
        self.hid2 = T.nn.Linear(38,76)
        self.oupt = T.nn.Linear(76, 1)
        self.relu = T.nn.ReLU()

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = T.tanh(self.hid1(x))
        z = self.relu(z)
        z = T.tanh(self.hid2(z))
        z = self.relu(z)
        z = T.sigmoid(self.oupt(z))
        return z


def accuracy(model, ds):
    # assumes model.eval()
    # granular but slow approach
    n_correct = 0
    n_wrong = 0
    for i in range(len(ds)):
        X = ds[i]['predictors']
        Y = ds[i]['targets']
        with T.no_grad():
            oupt = model(X)  # logits form
        oupt = oupt.numpy()
        max_index = oupt[0]
        if Y[0] == max_index:
            n_correct += 1
        else:
            n_wrong += 1

    acc = n_correct/(n_wrong+n_correct)
    return acc
# -----------------------------------------------------------

def main():
    # 0. get started
    base_dir = os.getcwd() + '/'
    file_dir = base_dir + "Msmarco/ance/"  # input folder, modify this if want to test on other dataset
    if not os.path.exists(file_dir+"model"):
        os.mkdir(file_dir+"model")
    print("\nBegin predict alpha value \n")
    np.random.seed(1)
    T.manual_seed(1)

    result_dict = {}
    for input_index in range(0, 11):
        doc = file_dir + 'eval/mrr10/' + str(input_index / 10) + '.eval'
        current_input = open(doc, 'r')
        current_lines = current_input.readlines()
        for current_line in current_lines:
            current_item = current_line.split()
            qid = current_item[1]
            current_score = current_item[2]
            if qid in result_dict.keys():
                previous = result_dict.get(qid)
                previous.append(current_score)
                result_dict[qid] = previous
            else:
                result_dict[qid] = [current_score]
    # 1. create DataLoader objects
    print("Creating train and test datasets ")
    test_file = file_dir + "feature_out/test.txt"
    testing_xdict = {}
    test_lines = open(test_file, 'r').readlines()

    for line in test_lines:
        items = line.split(sep='\t')
        xitem = [float(items[k]) for k in range(1, 20)]
        testing_xdict[items[0]] = T.tensor([xitem], dtype=T.float32)
    bat_size = 50
    #print(train_ldr)
    # 2. create network
    net = Net()#.to(device)

    # 3. train model
    max_epochs = 100
    ep_log_interval = 10
    lrn_rate = 0.01

    # -----------------------------------------------------------

    loss_func = T.nn.MSELoss()  # apply log-softmax()
    optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate, momentum=0.9)

    print("\nbat_size = %3d " % bat_size)
    print("loss = " + str(loss_func))
    print("optimizer = SGD")
    print("max_epochs = %3d " % max_epochs)
    print("lrn_rate = %0.3f " % lrn_rate)

    print("\nStarting train with saved checkpoints")

    net.train()
    max_line_length = 0

    train_file_name = file_dir + 'feature_out/train_ave.txt'


    for epoch in tqdm(range(0, max_epochs)):
        T.manual_seed(1 + epoch)  # recovery reproducibility
        epoch_loss = 0  # for one full epoch
        train_ds = InputDataset(train_file_name)
        train_ldr = T.utils.data.DataLoader(train_ds,
                                            batch_size=bat_size, shuffle=False)
        for (batch_idx, batch) in enumerate(train_ldr):
            X = batch['predictors']  # inputs
            Y = batch['targets']  # shape [10,3] (!)
            optimizer.zero_grad()

            oupt = net(X)  # shape [10] (!)
            loss_val = loss_func(oupt, Y)  # avg loss in batch

            epoch_loss += loss_val.item()  # a sum of averages

            loss_val.backward()

            optimizer.step()
        if epoch % ep_log_interval == 0:
            print("epoch = %4d   loss = %0.4f" % \
                  (epoch, epoch_loss))

            # checkpoint after 0-based epoch 100, 200, etc.

            print("Computing model accuracy")
            eval_results = []
            another = []
            maxi = []
            net.eval()
            acc_train = accuracy(net, train_ds)  # item-by-item
            print("Accuracy on training data = %0.4f" % acc_train)
            fn = file_dir + "model/model_bce_ave.pth"
            # output_model = open(fn, 'w')
            T.save(net.state_dict(), fn)
            for testing_element in testing_xdict.keys():
                inpt = testing_xdict.get(testing_element)
                with T.no_grad():
                    logits = net(inpt[0:19])
                logits = logits[0]
                max_index = int(logits*10)
                eval_result = float(result_dict.get(testing_element)[max_index])
                another_result = float(result_dict.get(testing_element)[2])
                maxi_reuslt = float(max(result_dict.get(testing_element)))
                eval_results.append(eval_result)
                another.append(another_result)
                maxi.append(maxi_reuslt)
            print("Predicted mrr10 value is : " + str(sum(eval_results) / len(eval_results)) + " vs " + str(
                sum(another) / len(another)) + " vs " + str(sum(maxi)/ len(maxi)))

    print("Training complete ")

    # 4. evaluate model accuracy
    #acc_test = accuracy(net, test_ds)  # en masse
    # acc_test = accuracy_quick(net, test_ds)  # en masse
    #print("Accuracy on test data = %0.4f" % acc_test)
    print("\nComputing model accuracy")
    net.eval()
    acc_train = accuracy(net, train_ds)  # item-by-item
    print("Accuracy on training data = %0.4f" % acc_train)

    # 5. make a prediction
    eval_results = []
    another = []
    maxi = []
    print("\nPredicting: ")
    count_p=0
    count_n=0
    count_e=0
    count_reach_m = 0
    count_not_reach_m = 0
    for testing_element in testing_xdict.keys():
        inpt = testing_xdict.get(testing_element)
        with T.no_grad():
            logits = net(inpt[0:19])  # values do not sum to 1.0
        logits = logits[0]
        max_index = int(logits*10)
        eval_result = float(result_dict.get(testing_element)[max_index])
        another_result = float(result_dict.get(testing_element)[2])
        maxi_reuslt = float(max(result_dict.get(testing_element)))
        if eval_result > another_result:
            count_p +=1
            print("The predicted appears at: " + str(max_index/10) + " with value of "+ result_dict.get(testing_element)[max_index] + " > fixed and max=" + max(result_dict.get(testing_element)))
        elif eval_result == another_result:
            count_e +=1
            print("The predicted appears at: " + str(max_index/10) + " with value of "+ result_dict.get(testing_element)[max_index] + " = fixed and max=" + max(result_dict.get(testing_element)))
        else:
            count_n +=1
            print("The predicted appears at: " + str(max_index / 10) + " with value of "+ result_dict.get(testing_element)[max_index] + " < fixed and max=" + max(result_dict.get(testing_element)))
        eval_results.append(eval_result)
        another.append(another_result)
        maxi.append(maxi_reuslt)
    print("Predicted mrr10 value is : " + str(sum(eval_results)/len(eval_results))  + " vs " + str(sum(another)/len(another)) + " vs " + str(sum(maxi)/ len(maxi)))
    ratio_p = count_p/(count_e+count_p+count_n)
    ratio_n = count_n/(count_e+count_p+count_n)
    ratio_e = 1-ratio_n-ratio_p
    print("\nHigher ratio = " + str(ratio_p) + " Equal ratio = " + str(ratio_e) + " Lower ratio = " + str(ratio_n) )
    # 6. save model (state_dict approach)
    print("\nSaving trained model ")
    fn = file_dir+ "model/model_bce_ave.pth"
    #output_model = open(fn, 'w')

    T.save(net.state_dict(), fn)

    print("\nEnd predict the output alpha value")


if __name__ == "__main__":
    main()

