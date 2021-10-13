from MixFormer import *
from loss import StructureLoss
from dataloader import AIHUBPedestrianDataset
from util_model import get_model
import importlib
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)

    args = parser.parse_args()

    cfg = importlib.import_module(
        os.path.splitext(os.path.basename(args.config))[0]
        )


    device = torch.device('cuda') if cfg.device == 'cuda' else torch.device('cpu')

    dataset = AIHUBPedestrianDataset(cfg.dir_image, cfg.imgh, cfg.imgw, cfg.augmentation)
    ###
    batch_size = cfg.num_class if cfg.batch_size > cfg.num_class else cfg.batch_size
    steps_per_epoch = int(dataset.size() / batch_size)
    max_epoch = int(cfg.max_epoch * dataset.size() / len(dataset))
    ###

    loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=2)

    net, dim_feature = get_model(cfg.model, cfg.num_class)
    net.to(device)

    optimizer = torch.optim.AdamW(net.parameters(), cfg.lr)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, cfg.lr, epochs=max_epoch, steps_per_epoch=steps_per_epoch, 
        pct_start = 0.1, anneal_strategy='linear', final_div_factor=1e2)

    loss_fn_metric = StructureLoss(cfg.num_class, dim_feature, device)
    loss_fn_softmax = nn.CrossEntropyLoss()

    for i in range(max_epoch):

        for data, label in loader:
            optimizer.zero_grad()

            data = data.permute(0, 3, 1, 2).to(device)
            label = label.to(device)

            feature, logit = net(data)
            loss_center, loss_push, loss_gpush = loss_fn_metric(feature, label)
            loss_softmax = loss_fn_softmax(logit, label)
            loss = loss_center + loss_push + loss_gpush + loss_softmax

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            print('-' * 50)
            print('Epoch %.2f' % float(1 + len(dataset) * i / dataset.size()))
            print('LR :', '%e' % lr_scheduler.get_last_lr()[0])
            print('center :', loss_center.item())
            print('push :', loss_push.item())
            print('gpush :', loss_gpush.item())
            print('softmax :', loss_softmax.item())

    torch.save(net.state_dict(), cfg.save_path)


    



