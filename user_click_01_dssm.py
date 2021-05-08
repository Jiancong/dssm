import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
#from dnn_model import DNN
#from encoder_model import Encoder
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from weight_initializer import Initializer
import torch.nn.init as init
import time
import os
import faiss
import timeit
from datetime import date, timedelta

from sklearn import preprocessing

from pynvml import *

from fileinput import input
from glob import glob 

def show_gpu_memory():
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {(info.total/1024)/1024} MB')
    print(f'free     : {(info.free/1024)/1024} MB')
    print(f'used     : {(info.used/1024)/1024} MB')

class DNNModel(nn.Module):
    def __init__(self, input_user, input_item, out, input_user_categorical_feature, input_item_categorical_feature, 
                 hidden_layers, dropouts, batch_norm):
        super(DNNModel, self).__init__()
        # 6040,  128
        # 3883,  128
        self.user_embed = nn.Embedding(input_user_categorical_feature[0][0], input_user_categorical_feature[0][1])
        self.item_embed = nn.Embedding(input_item_categorical_feature[0][0], input_item_categorical_feature[0][1])
        
        self.user_dnn = nn.Sequential(
            nn.Linear(input_user, out),
            nn.LeakyReLU(),
            nn.Linear(out, out),
            nn.LeakyReLU()
        )
        
        self.item_dnn = nn.Sequential(
            nn.Linear(input_item, out),
            nn.LeakyReLU(),
            nn.Linear(out, out),
            nn.LeakyReLU()
        )
        
    def forward(self, x):
        user = self.user_embed(x[:, 0])
        item = self.item_embed(x[:, 1])

        user = self.user_dnn(user)
        item = self.item_dnn(item)

        user = user/torch.sum(user*user, 1).view(-1,1)

        item = item/torch.sum(item*item, 1).view(-1,1)

        return user, item

class trainset(Dataset):
    def __init__(self, data):
        self.x= data[0]
        self.y= data[1]

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        data = (x, y)
        return data

    def __len__(self):
        return len(self.x)

def random_sample_01_method(train_data):

    # 随机负采样
    # 采样3个样本
    sample_list = list(train_data['movie'].unique())
    
    # data变成了[user, mov, 1/0]的列表
    data = list()
    
    for idx, rows in tqdm(train_data.iterrows(), total=len(train_data)):
        use = rows['user']
        mov = rows['movie']
        # 添加正例
        data.append([use, mov, 1])
        
        for m in np.random.choice(sample_list, 3):
            # 添加负例
            data.append([use, m, 0])
        
    data = pd.DataFrame(data, columns=['user', 'movie', 'tag']) 
    return data

def train_model(model, train_loader, val_loader, epoch, loss_function, optimizer, early_stop, minloss_model_filename):
    """
    pytorch 模型训练通用代码
    :param model: pytorch 模型
    :param train_loader: dataloader, 训练数据
    :param val_loader: dataloader, 验证数据
    :param epoch: int, 训练迭代次数
    :param loss_function: 优化损失函数
    :param optimizer: pytorch优化器
    :param early_stop: int, 提前停止步数
    :param best_auc_model_filename: string, 最好auc的state_dict参数存储文件名
    :param best_val_auc: float, 最好的validation auc分数
    :return: None
    """
    print("Entering train_model....")
    # 是否使用GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_val_auc = 0.0

    print("device is ready....")
    
#     device = torch.device("cpu")

    model = model.to(device)

    print ("model to device done....")
    
    # 多少步内验证集的loss没有变小就提前停止
    patience, eval_loss = 0, 0
    
    # 训练
    for i in range(epoch):
        print("Running epoch : {} pass.".format(i))
        total_loss, count = 0, 0
        # 预测值
        y_pred = list()
        # ground_truth 值
        y_true = list()
        
        for idx, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            user, item = model.forward(x)

            predict = torch.sigmoid(torch.sum(user*item, 1))

            y_pred.extend(predict.cpu().detach().numpy())
            y_true.extend(y.cpu().detach().numpy())

            loss = loss_function(predict.squeeze(), y.float().squeeze())
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            count += 1
            
        train_auc = roc_auc_score(np.array(y_true), np.array(y_pred))
        print("Epoch %d train loss is %.3f and train auc is %.3f" % (i+1, total_loss / count, train_auc))
    
        # 验证
        total_eval_loss = 0
        model.eval()
        count_eval = 0
        val_y_pred = list()
        val_true = list()
        for idx, (x, y) in tqdm(enumerate(val_loader), total=len(val_loader)):
            x, y = x.to(device), y.to(device)
            u, m = model(x)
            predict = torch.sigmoid(torch.sum(u*m, 1))
            val_y_pred.extend(predict.cpu().detach().numpy())
            val_true.extend(y.cpu().detach().numpy())
            loss = loss_function(predict.squeeze(), y.float().squeeze())
            total_eval_loss += float(loss)
            count_eval += 1
        val_auc = roc_auc_score(np.array(y_true), np.array(y_pred))
        print("Epoch %d val loss is %.3f and val auc is %.3f" % (i+1, total_eval_loss / count_eval, val_auc))

        if val_auc > best_val_auc : 
            print("val_auc:{} is best than {} ".format(val_auc, best_val_auc))
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'model/userclick_best_val_auc_model.pt')
        
        # 提前停止策略
        if i == 0:
            eval_loss = total_eval_loss / count_eval
        else:
            if total_eval_loss / count_eval < eval_loss:
                eval_loss = total_eval_loss / count_eval
                print ("new min loss, save the model.")
                torch.save(model.state_dict(), minloss_model_filename)
            else:
                if patience < early_stop:
                    patience += 1
                else:
                    print("val loss is not decrease in %d epoch and break training" % patience)
                    break

if __name__ == "__main__":
    show_gpu_memory()

    best_val_auc = 0.0

    best_model_file = "model/userclick_min_loss_model.pt"

    startdate = date(2021,4,28)
    enddate = date(2021,4,29)

    train_data = pd.DataFrame()
    test_data= pd.DataFrame()

    li = []

    ROWS_NUM = 1000000


    test_data_dir = "data/dt=2021-04-29/gen_1day_samples/"
    for file_name in glob(test_data_dir + "00000*"):
        print("retrieve test file:{}".format(file_name))
        df = pd.read_csv(file_name, sep = ",", index_col=None, header=0, nrows = ROWS_NUM)
        df.columns=['user', 'item', 'tag', 'count']
        li.append(df)

    test_data = pd.concat(li, axis=0, ignore_index=True)

    li = []

    print("test data shape: {}".format(test_data.shape))
    print("test data done....")

    for dt in pd.date_range(startdate,enddate-timedelta(days=1),freq='d'):
 
        sample_data_dir = "data/" + "dt=" + dt.strftime('%Y-%m-%d') + "/gen_1day_samples/"
        #sample_data_dir = "data/"

        root_path = os.path.abspath('./')
        for file_name in glob(sample_data_dir+'00000*'):
            #print("retrieve train file:{}".format(file_name))
            df = pd.read_csv(file_name, sep=',', index_col=None, header=0, nrows = ROWS_NUM)
            df.columns=['user', 'item', 'tag', 'count']
            li.append(df)

    train_data = pd.concat(li, axis=0, ignore_index=True)

    # 将click_num > 1的处理成0,1二值
    train_data.loc[train_data['tag'] > 1, 'tag'] = 1

    print("Processing sample data....")

    print("data shape: {}".format(train_data.shape))

    #train_data.columns = ['user', 'item', 'tag']

    user_le = preprocessing.LabelEncoder()
    item_le = preprocessing.LabelEncoder()
    tag_le = preprocessing.LabelEncoder()
    # convert 'user'
    train_data_user = user_le.fit_transform(train_data[['user']].values) 
    train_data_user_num = len(user_le.classes_)
    print("user classes num: {}".format(len(user_le.classes_)))

    # convert 'item'
    train_data_item = item_le.fit_transform(train_data[['item']].values) 
    train_data_item_num = len(item_le.classes_)

    print("item classes num: {}".format(len(item_le.classes_)))
    # convert 'tag'
    train_data_tag = tag_le.fit_transform(train_data[['tag']].values) 
    print("tag classes num: {}".format(len(tag_le.classes_)))
    print("tag classes : {}".format((tag_le.classes_)))
    print("tag inverse classes:{}".format(tag_le.inverse_transform(tag_le.classes_)))

    train_data_labeled = pd.DataFrame()

    train_data_labeled['user'] = train_data_user.tolist()
    train_data_labeled['item'] = train_data_item.tolist()
    train_data_labeled['tag'] = train_data_tag.tolist()

    train_df, val_df = train_test_split(train_data_labeled, test_size=0.2, random_state=2021)

    print ("train_df.shape:{}".format(train_df.shape))
    print ("val_df.shape:{} ".format(val_df.shape ))

    # 训练集验证集随机分割
    train_x_tensor = torch.tensor(train_df[['user', 'item']].values)
    train_y_tensor = torch.tensor(train_df[['tag']].values)
    val_x_tensor = torch.as_tensor(val_df[['user', 'item']].values)
    val_y_tensor = torch.as_tensor(val_df[['tag']].values)
    
    # 构造dataloader
    train_dataset = trainset((train_x_tensor, train_y_tensor))
    val_dataset = trainset((val_x_tensor, val_y_tensor))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # 模型初始化
    input_user = 128
    input_item = 128
    out = 64
    # 
    input_user_categorical_feature = {0: (train_data_user_num, 128)}
    input_item_categorical_feature =  {0: (train_data_item_num, 128)}

    hidden_layers = [128, 64]
    dropouts = [0.5, 0.5, 0.5]
    batch_norm = False

    print("Preparing to train model.....")
    
    model = DNNModel(input_user, input_item, out, input_user_categorical_feature, input_item_categorical_feature, 
                     hidden_layers, dropouts, batch_norm)
    Initializer.initialize(model=model, initialization=init.xavier_uniform, gain=init.calculate_gain('relu'))
    
    # 模型训练
    epoch = 20
    loss_function = F.binary_cross_entropy_with_logits
    early_stop = 3
    learn_rate = 5e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    show_gpu_memory()

    t0 = timeit.default_timer()
    
    if not os.path.isfile(best_model_file) :
        print("Training ....")
        train_model(model, train_loader, val_loader, epoch, loss_function, optimizer, early_stop, best_model_file) 
    else:
        print("Load pre-trained model.....")
        model.load_state_dict(torch.load(best_model_file))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

    t1 = timeit.default_timer()
    torch.cuda.synchronize()
    print("Cuda Synch takes {:.2f}".format(timeit.default_timer()-t1))

    print("Model to device takes {:.2f}".format(timeit.default_timer()-t0))

    print("Evaluating .....")

    #print("user info:{}".format(user.shape))

    # 结果验证
    if 0:
        model.eval()
        user['movie'] = 1
        test_x = user[['user', 'movie']].values
        x = torch.from_numpy(test_x).cuda()
        user_embed, _ = model(x)
        
        movie['user'] = 1
        test_x = movie[['user', 'movie']].values
        x = torch.from_numpy(test_x).cuda()
        _, movie_embed = model(x)
        
        movie_embed = movie_embed.cpu().detach().numpy()
        user_embed = user_embed.cpu().detach().numpy()
        

        ## faiss索引构建
        d = 64
        nlist = 10
        index = faiss.IndexFlatL2(d)
        index.add(movie_embed)
        
        # 验证集数据字典化
        user_movie_dict_val = dict()
        for idx, rows in tqdm(val_data.iterrows(), total=len(val_data)):
            u = rows['user']
            m = rows['movie']
            if u not in user_movie_dict_val:
                user_movie_dict_val[u] = [m]
            else:
                user_movie_dict_val[u].append(m)
                    
        # 用户推荐结果索引           
        D, I = index.search(user_embed[list(val_data['user'].unique())], 50)
        
        # 召回率计算
        hits, total, total1 = 0, 0, 0
        for uid, rec_list in zip(list(val_data['user'].unique()), I):
            hits += len(set(rec_list)&set(user_movie_dict_val[uid]))
            total += len(user_movie_dict_val[uid])
            total1 += len(set(rec_list))

        # 率计算
        recall = hits/total
        precision = hits/total1
        F1 = 2*recall*precision/(recall+precision)
        print("recall is %.3f" % (hits/total))
        print("precision is %.3f" % (hits/total1))
        print("F1 score is %.3f" % (F1))
