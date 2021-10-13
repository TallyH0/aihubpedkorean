from MixFormer import *
from dataloader import AIHUBPedestrianDataset
from dataloader import Market1501PedestrianDataset
import importlib
import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from util_model import get_model

def evalute(discriptor, labels):
    num_discriptor = discriptor.shape[0]
    rank1_acc = 0
    ap = 0
    
    for i in range(num_discriptor):
        score_matrix = np.matmul(discriptor[i], discriptor.transpose())
        second_max_index = np.argsort(score_matrix)[-2]
        
        id_src = labels[i]
        id_rank1 = labels[second_max_index]
        
        y_true = np.array(id_src == labels, np.int32)
        ap += roc_auc_score(y_true, score_matrix, average='weighted')
        
        if id_src == id_rank1:
            rank1_acc += 1
            
    return rank1_acc / len(labels), ap / len(labels)

def evalute_market1501(discriptor_query, labels_query, discriptor_test, labels_test):
    rank1_acc = 0
    ap = 0
    
    score_matrix = np.matmul(discriptor_query, discriptor_test.transpose())
    
    for i in range(len(score_matrix)):
        argmax = np.argmax(score_matrix[i])
        if labels_query[i] == labels_test[argmax]:
            rank1_acc += 1
            
        y_true = np.array(labels_query[i] == labels_test, np.int32)
        ap += roc_auc_score(y_true, score_matrix[i])
    
            
    return rank1_acc / len(labels_query), ap / len(labels_query)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dir_query', type=str)
    parser.add_argument('--dir_test', type=str)

    args = parser.parse_args()

    cfg = importlib.import_module(
        os.path.splitext(os.path.basename(args.config))[0]
        )

    device = torch.device('cuda') if cfg.device == 'cuda' else torch.device('cpu')
    net, dim_feature = get_model(cfg.model, cfg.num_class)
    net.to(device)
    weight_dict = torch.load(cfg.save_path)
    net.load_state_dict(weight_dict)

    if args.dataset =='AIHUB':
        dataset = AIHUBPedestrianDataset(cfg.dir_test , cfg.imgh, cfg.imgw, transform=None, train=False)
        loader = torch.utils.data.DataLoader(dataset, cfg.batch_size//2, shuffle=True, num_workers=2)

        discriptor = []
        labels = []

        net.eval()
        i = 0
        for data, label in loader:
            print(i, len(loader))
            i += 1

            data = data.permute(0, 3, 1, 2).to(device)
            label = label.detach().numpy()

            features, logit = net(data)
            features = F.normalize(features)

            features = features.detach().to(torch.device('cpu')).numpy()
            for feature in features:
                discriptor.append(feature)
            for id in label:
                labels.append(id)
        
        discriptor = np.array(discriptor)
        rank1, mAP = evalute(discriptor, labels)
        print('AIHUB Validation')
        print('Rank1 Acc:', rank1)
        print('mAP :', mAP)

    elif args.dataset == 'Market1501':

        if args.dir_query and args.dir_test:

            dataset_query = Market1501PedestrianDataset(args.dir_query, cfg.imgh, cfg.imgw, transform=None, train=False)
            dataset_test = Market1501PedestrianDataset(args.dir_test, cfg.imgh, cfg.imgw, transform=None, train=False)
            loader_query = torch.utils.data.DataLoader(dataset_query, 16)
            loader_test = torch.utils.data.DataLoader(dataset_test, 16)

            discriptor_query = []
            discriptor_test = []
            labels_query = []
            labels_test = []

            net.eval()
            i = 0
            for data, label in loader_query:

                data = data.permute(0, 3, 1, 2).to(device)
                label = label.detach().numpy()

                features, logit = net(data)
                features = F.normalize(features)

                features = features.detach().to(torch.device('cpu')).numpy()
                for feature in features:
                    discriptor_query.append(feature)
                for id in label:
                    labels_query.append(id)
                    
            for data, label in loader_test:

                data = data.permute(0, 3, 1, 2).to(device)
                label = label.detach().numpy()

                features, logit = net(data)
                features = F.normalize(features)

                features = features.detach().to(torch.device('cpu')).numpy()
                for feature in features:
                    discriptor_test.append(feature)
                for id in label:
                    labels_test.append(id)
                    
            discriptor_query = np.array(discriptor_query)
            discriptor_test = np.array(discriptor_test)

            rank1, mAP = evalute_market1501(discriptor_query, labels_query, discriptor_test, labels_test)
            print('Market1501 test protocol')
            print('Rank1 Acc:', rank1)
            print('mAP :', mAP)

        else:
            print('For Market1501, please add argument --dir_query, --dir_test')
            exit(-2)
    else:
        print('Wrong dataset, please type AIHUB or Market1501')
        exit(-1)
