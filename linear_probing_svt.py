from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
import argparse
from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('train_feature_path', type=str)
    parser.add_argument('test_feature_path', type=str)
    parser.add_argument('train_label_path', type=str, default="/data/ahngeo11/svt/datasets/annotations/kth_fe_videos.txt")
    parser.add_argument('test_label_path', type=str)
    parser.add_argument('--feature_dim', type=int)
    parser.add_argument('--num_labels', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int, default=20)
    
# python linear_probing_svt.py /data/ahngeo11/svt/outputs/diving48/trainfeat.pkl /data/ahngeo11/svt/outputs/diving48/testfeat.pkl /data/ahngeo11/svt/datasets/annotations/diving48_train_videos.txt /data/ahngeo11/svt/datasets/annotations/diving48_test_videos.txt --feature_dim 768 --num_labels 48 --lr 5e-4
    
    args = parser.parse_args()

    with open(args.train_feature_path, 'rb') as f :
        data = pickle.load(f)
        X_train = np.array(data)

    with open(args.test_feature_path, 'rb') as f :
        data = pickle.load(f)
        X_val = np.array(data)
            
    with open(args.train_label_path, 'r') as f :
        label = f.readlines()
        label = [label[i].strip('\n').split()[1] for i in range(len(label))]
        y_train = np.array(label)
    
    with open(args.test_label_path, 'r') as f :
        label = f.readlines()
        label = [label[i].strip('\n').split()[1] for i in range(len(label))]
        y_val = np.array(label)
        
    X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(X_val.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32)), torch.from_numpy(y_val.astype(np.float32))
    
    dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
    dataset_val = TensorDataset(X_val_tensor, y_val_tensor)
    
    # dataset_train = TensorDataset(X_train, y_train)
    # dataset_val = TensorDataset(X_val, y_val)
    
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=True)
    
    linear_classifier = LinearClassifier(args.feature_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    
    # linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])    
    
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        args.lr,
        momentum=0.9,
        # weight_decay=0,
        weight_decay=0.0001) # we do not apply weight decay)
        
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[11,14], gamma=0.1)
      
    
    linear_classifier.train()
    for epoch in range(args.epochs) :
        for batch_x, batch_y in dataloader_train :
            batch_x = batch_x.cuda()
            batch_y = batch_y.type(torch.LongTensor).cuda()
            
            output = linear_classifier(batch_x)
            
            loss = nn.CrossEntropyLoss()(output, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            
            print("epoch " + str(epoch) + ": " + str(loss.item()))
            
    
    score = 0
    total = 0
    linear_classifier.eval()
    for batch_x, batch_y in dataloader_val :
        batch_x = batch_x.cuda()
        batch_y = batch_y.type(torch.LongTensor).cuda()
        
        with torch.no_grad():
            output = linear_classifier(batch_x)
            loss = nn.CrossEntropyLoss()(output, batch_y)
            print("val loss : ", str(loss.item()))
            
            pred = torch.argmax(output, dim=1)
            correct_pred = (batch_y == pred)
            score += correct_pred.tolist().count(True)
            total += len(batch_x)
    
    print("linear acc : ", score / total)

    
    ###################################################################
    knn = KNeighborsClassifier(n_neighbors=3, weights="distance")
    knn.fit(X_train, y_train)

    knn_pred = knn.predict(X_val)

    knn_eval = (knn_pred == y_val).tolist()
    
    knn_acc = knn_eval.count(True) / len(knn_eval)

    print("knn acc : ", knn_acc)

