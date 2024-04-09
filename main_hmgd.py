import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time


from global_loss import Global_Loss
from model import *
from utils import *
from evaluation import *



def train(model, optimizer, scheduler, trainloader, testloader, tree, CELoss, GLoss, args):
    device = args['device']
    alpha = args['alpha']
    # beta = args['beta']
    model = model.to(device)
    InternalNodes = tree_InternalNodes(tree)
    root = tree_Root(tree)
    noLeafNode = np.append(InternalNodes, root)
    LeafNode = tree_LeafNode(tree)
    # records
    train_loss = []
    train_ce_loss = []
    train_g_loss = []
    train_time = []
    # RkLoss = RankLoss(correlation, device)
    minloss = 1e10
    max_acc = 0
    max_FH = 0

    for epoch in range(args['epochs']):
        start = time.time()
        loss_sum = 0
        celoss_sum = 0
        gloss_sum = 0
        model.train()
        scheduler.step()
        # optimizer.param_groups[0]['lr'] = cosine_anneal_schedule(epoch, args['epochs'], args['learning_rate'])
        batch_num = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch_num += 1
            inputs, targets = inputs.to(torch.float32).to(device), targets.to(torch.long).to(device)

            re_idx = np.where(np.in1d(targets.cpu(), noLeafNode))[0]
            nore_idx = np.where(np.in1d(targets.cpu(), LeafNode))[0]
            # forward
            out, final_out = model(inputs, device)
            mul_out = final_out
            for i in range(model.level_num-2, -1, -1):
                mul_out = torch.cat((mul_out, out[i]), dim=1)
            if re_idx.size != 0:
                mul_out[re_idx, 0: model.n_out[-1]] = 0
            # loss
            ce_loss = 0
            tree_gap = model.n_out[-1]
            cur_targets_ori = targets.cpu()
            temp_targets = torch.tensor(targets)
            temp_targets[re_idx] = 0
            mul_label = F.one_hot(temp_targets, num_classes=model.n_out[-1])
            mul_label[re_idx, 0] = 0
            for i in range(model.level_num-2, -1, -1):
                if i == model.level_num - 2 and re_idx.size != 0:
                    cur_targets_ori[nore_idx] = torch.tensor(tree[0, cur_targets_ori[nore_idx]]).to(torch.long)
                else:
                    cur_targets_ori = tree[0, cur_targets_ori]
                cur_targets = torch.tensor(cur_targets_ori - tree_gap).to(torch.long).to(device)
                tree_gap += model.n_out[i]
                if cur_targets.size() == torch.Size([]):
                    cur_targets = torch.unsqueeze(cur_targets, dim=0)
                # try:
                mul_label = torch.cat((mul_label, F.one_hot(cur_targets, num_classes=model.n_out[i])), dim=1)
                ce_loss += CELoss(out[i], cur_targets)

            ce_loss += CELoss(final_out[nore_idx], targets[nore_idx])
            ce_loss /= model.level_num
            g_loss = GLoss.forward(mul_out, mul_label, targets[nore_idx])
            # rk_loss = RkLoss(out, targets, noLeafNode)
            loss = (1-alpha) * ce_loss + alpha * g_loss
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            celoss_sum += ce_loss.item()
            gloss_sum += g_loss.item()

        end = time.time()
        # record
        train_loss.append(loss_sum/batch_num)
        train_ce_loss.append(celoss_sum/batch_num)
        train_g_loss.append(gloss_sum/batch_num)
        train_time.append(end-start)

        if epoch % 1 == 0:
            print('Epoch {:04d}: '.format(epoch + 1))
            print('loss: {:.03f} '.format(train_loss[-1]),
                  'ce_loss: {:.03f} '.format(train_ce_loss[-1]),
                  'g_loss: {:.03f} '.format(train_g_loss[-1]),
                  # 'mean_time: {:.05f} '.format(np.mean(train_time[-10:len(train_time)])),
                  # 'test_acc: {:.05f}'.format(acc),
                  'time: {:.05f} '.format(train_time[-1]),
                  )
            acc, FH, TIE, FLCA = test(model, testloader, CELoss, GLoss, tree, args)
            if acc > max_acc:
                max_acc = acc
                max_epoch = epoch+1
                match_FH = FH
                match_TIE =TIE
                match_FLCA = FLCA
                torch.save(model,
                           './results/model_' + args['dataset'] + '_lr' + str(args['learning_rate']) + '_alpha' + str(args['alpha']) + '_re' + str(args['relabel']) + '.pth')
            print("max_acc:{:.02f}".format(max_acc),
                  'match_FH:{:.02f}'.format(match_FH),
                  'match_TIE:{:.02f}'.format(match_TIE),
                  'match_FLCA:{:.02f}'.format(match_FLCA),
                  'max_epoch:', max_epoch
                  )

def test(model, testloader, CELoss, GLoss, tree, args):
    device = args['device']
    # correct = 0
    # total = 0
    test_loss = 0
    alpha = args['alpha']
    # beta = args['beta']
    # train_ce_loss = []
    labels_pre = torch.empty(0).to(torch.long).to(device)
    labels = torch.empty(0).to(torch.long).to(device)
    with torch.no_grad():
        model.eval()
        start = time.time()
        ce_loss = 0
        g_loss = 0
        batch_num = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            batch_num += 1
            inputs, targets = inputs.to(torch.float32).to(device), targets.to(torch.long).to(device)
            out, final_out = model(inputs, device)
            mul_out = final_out
            for i in range(model.level_num-2, -1, -1):
                mul_out = torch.cat((mul_out, out[i]), dim=1)
            tree_gap = model.n_out[-1]
            cur_targets_ori = targets.cpu()
            mul_label = F.one_hot(targets, num_classes=model.n_out[-1])
            for i in range(model.level_num - 2, -1, -1):
                cur_targets_ori = tree[0, cur_targets_ori]
                cur_targets = torch.tensor(cur_targets_ori - tree_gap).to(torch.long).to(device)
                tree_gap += model.n_out[i]
                if cur_targets.size() == torch.Size([]):
                    cur_targets = torch.unsqueeze(cur_targets, dim=0)
                mul_label = torch.cat((mul_label, F.one_hot(cur_targets, num_classes=model.n_out[i])), dim=1)
                ce_loss += CELoss(out[i], cur_targets)
            ce_loss += CELoss(final_out, targets)
            g_loss += GLoss.forward(mul_out, mul_label, targets)

            # labels_pre.extend(torch.argmax(final_out, dim=1).cpu().numpy().tolist())
            # labels.extend(targets.cpu().numpy().tolist())
            labels_pre = torch.cat((labels_pre, torch.argmax(final_out, dim=1)), dim=0)
            labels = torch.cat((labels, targets), dim=0)
            # correct += (torch.argmax(final_out, dim=1) == targets).sum()
            # total += targets.shape[0]
        test_time = time.time() - start
        ce_loss /= model.level_num
        test_loss = (1-alpha) * ce_loss.item() + alpha * g_loss.item()
        # acc = 100. * correct / total
        acc = 100. * labels_pre.eq(labels).cpu().sum().item()/len(labels)
        FH = 100. * EvaHier_HierarchicalPrecisionAndRecall(labels_pre.cpu().numpy(), labels.cpu().numpy(), tree)
        TIE = 100. * EvaHier_TreeInducedError(labels_pre.cpu().numpy(), labels.cpu().numpy(), tree)
        FLCA = 100. * EvaHier_HierarchicalLCAPrecisionAndRecall(labels_pre.cpu().numpy(), labels.cpu().numpy(), tree)
        print('test_loss: {:.03f} '.format(test_loss/batch_num),
              'ce_loss: {:.03f} '.format(ce_loss.item()/batch_num),
              'g_loss: {:.03f} '.format(g_loss.item()/batch_num),
              # 'mean_time: {:.05f} '.format(np.mean(train_time[-10:len(train_time)])),
              'test_acc: {:.04f}'.format(acc),
              'test_FH: {:.04f}'.format(FH),
              'test_TIE: {:.04f}'.format(TIE),
              'test_FLCA: {:.04f}'.format(FLCA),
              'time: {:.04f} '.format(test_time),
              )
    return acc, FH, TIE, FLCA



def main(args):
    device = torch.device('cuda:' + str(args['gpu']) if torch.cuda.is_available() else 'cpu')
    args['device'] = device
    setup_seed(args['seed'])
    # create data
    if args['dataset_id'] == 1:
        pre = True
    else:
        pre = False
    train_data = MLLDataset(args, istrain=True, pre=pre)
    test_data = MLLDataset(args, istrain=False, pre=pre)
    features = train_data.features
    labels = train_data.labels
    tree = train_data.tree
    InternalNodes = tree_InternalNodes(tree)
    root = tree_Root(tree)
    noLeafNode = np.append(InternalNodes, root)

    n_feature = features.shape[1]
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args['batch_size'], shuffle=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args['batch_size'], shuffle=False, drop_last=False)

    model = HLE(n_in=n_feature, n_hidden=args['n_hidden'], tree=tree,
                keep_prob=args['keep_prob'])

    CELoss = nn.CrossEntropyLoss()
    GLoss = Global_Loss(tree, model.n_out, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=1e-5)
    if args['dataset_id'] == 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.2, last_epoch=-1)
    elif args['dataset_id'] == 1:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.2, last_epoch=-1)
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[2, 5], gamma=0.2, last_epoch=-1)

    print('Begin training pharse')
    train(model, optimizer, scheduler, trainloader, testloader, tree, CELoss, GLoss, args)

    model = torch.load('./results/model_' + args['dataset'] + '_lr' + str(args['learning_rate']) + '_alpha' + str(args['alpha']) + '_re' + str(args['relabel']) + '.pth').to(
        device)
    acc = test(model, testloader, CELoss, GLoss, tree, args)
