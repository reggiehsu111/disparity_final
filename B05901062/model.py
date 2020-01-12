import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
from pathlib import Path
from util import readPFM
import time

class CNN(nn.Module):
    def __init__(self, fmaps, kernel_size):
        self.fmaps = fmaps
        self.kernel_size = kernel_size

        super(CNN, self).__init__()
        self.conv1_1 = nn.Conv2d(1, self.fmaps, self.kernel_size)
        self.conv1_2 = nn.Conv2d(self.fmaps, self.fmaps, self.kernel_size)
        self.conv1_3 = nn.Conv2d(self.fmaps, self.fmaps, self.kernel_size)
        self.conv1_4 = nn.Conv2d(self.fmaps, self.fmaps, self.kernel_size)
        self.conv1_5 = nn.Conv2d(self.fmaps, self.fmaps, self.kernel_size)

        self.conv2_1 = nn.Conv2d(1, self.fmaps, self.kernel_size)
        self.conv2_2 = nn.Conv2d(self.fmaps, self.fmaps, self.kernel_size)
        self.conv2_3 = nn.Conv2d(self.fmaps, self.fmaps, self.kernel_size)
        self.conv2_4 = nn.Conv2d(self.fmaps, self.fmaps, self.kernel_size)
        self.conv2_5 = nn.Conv2d(self.fmaps, self.fmaps, self.kernel_size)
    
    def forward(self, x, y):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.relu(self.conv1_3(x))
        x = F.relu(self.conv1_4(x))
        x = F.relu(self.conv1_5(x))

        y = F.relu(self.conv2_1(y))
        y = F.relu(self.conv2_2(y))
        y = F.relu(self.conv2_3(y))
        y = F.relu(self.conv2_4(y))
        y = F.relu(self.conv2_5(y))

        return x, y

class FCNet(nn.Module):
    def __init__(self, fmaps, fc_units):
        self.fmaps = fmaps
        self.fc_units = fc_units

        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(self.fmaps * 2, self.fc_units)
        self.fc2 = nn.Linear(self.fc_units, self.fc_units)
        self.fc3 = nn.Linear(self.fc_units, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = self.sigmoid(self.fc3(z))
        return z

class Model():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.data_neg_low = 1.5
        self.data_neg_high = 18
        self.data_pos = 0.5
        self.patch_size = 11
        self.fmaps = 112
        self.kernel_size = 3
        self.fc_units = 384
        self.cnn = CNN(self.fmaps, self.kernel_size).to(self.device)
        self.fcnet = FCNet(self.fmaps, self.fc_units).to(self.device)
        self.lr = 0.003
        self.momentum = 0.9
        self.batch_size = 256
        self.epochs = 14
    
    def train(self, data_path):
        self.__preprocess(data_path)

        criterion = nn.BCELoss()

        optimizer = optim.SGD(
            [{'params': self.cnn.parameters()}, {'params': self.fcnet.parameters()}],
            lr=self.lr,
            momentum=self.momentum
        )

        trainloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                self.train_pairs,
                self.train_labels
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

        for epoch in range(self.epochs):
            since = time.time()

            running_loss = 0.0
            running_corrects = 0
            for i, data in enumerate(trainloader):
                inputs, labels = data

                patches_l = []
                patches_r = []
                for [p, q, idx] in inputs:
                    patches_l.append(
                        torch.unsqueeze(torch.unsqueeze(torch.from_numpy(self.train_imgs_l[
                            idx[0]][
                            p[0]: p[0] + self.patch_size,
                            p[1]: p[1] + self.patch_size
                        ]), 0), 0)
                    )
                    patches_r.append(
                        torch.unsqueeze(torch.unsqueeze(torch.from_numpy(self.train_imgs_r[
                            idx[0]][
                            q[0]: q[0] + self.patch_size,
                            q[1]: q[1] + self.patch_size
                        ]), 0), 0)
                    )

                labels = labels.to(self.device)
                patches_l = torch.cat(patches_l).float().to(self.device)
                patches_r = torch.cat(patches_r).float().to(self.device)

                optimizer.zero_grad()

                x, y = self.cnn(patches_l, patches_r)
                z = torch.cat((x.view(-1, self.fmaps), y.view(-1, self.fmaps)), 1)
                outputs = self.fcnet(z)

                preds = torch.round(outputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # print(loss.item(), torch.sum(preds == labels).float() / inputs.size(0))
                running_corrects += torch.sum(preds == labels).float() / inputs.size(0)

                if i % 2000 == 1999:
                    time_elapsed = time.time() - since
                    print('[%d, %5d] loss: %.3f, acc: %.3f, time: %.0fm %.0fs' %
                        (epoch + 1, i + 1, running_loss / 2000, running_corrects / 2000, time_elapsed // 60, time_elapsed % 60))
                    running_loss = 0.0
                    running_corrects = 0
                
            torch.save(self.cnn.state_dict(), 'model/cnn_e:{}.pkl'.format(epoch))
            torch.save(self.fcnet.state_dict(), 'model/fcnet_e:{}.pkl'.format(epoch))
        print('Finished Training')
    
    def load_model(self, cnn_model_path, fcnet_model_path):
        self.cnn.load_state_dict(torch.load(cnn_model_path, map_location=self.device))
        self.fcnet.load_state_dict(torch.load(fcnet_model_path, map_location=self.device))

    def __calc_features(self, img_l, img_r):
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        half_patch_size = self.patch_size // 2
        img_l = F.pad(torch.from_numpy(img_l).float(),
            (half_patch_size, half_patch_size, half_patch_size, half_patch_size))
        img_r = F.pad(torch.from_numpy(img_r).float(),
            (half_patch_size, half_patch_size, half_patch_size, half_patch_size))

        for i in range(2):
            img_l = torch.unsqueeze(img_l, 0)
            img_r = torch.unsqueeze(img_r, 0)
        img_l = img_l.to(self.device)
        img_r = img_r.to(self.device)

        features_l, features_r = self.cnn(img_l, img_r)
        return features_l, features_r

    def calc_l_similarity(self, img_l, img_r, d_max):
        features_l, features_r = self.__calc_features(img_l, img_r)

        features_l = torch.squeeze(features_l)
        features_r = torch.squeeze(features_r)

        _, h, w = features_l.size()
        d_max -= 1
        outputs = torch.empty(h, w, d_max + 1)
        for i in range(h):
            for j in range(w):
                d_range = min(d_max, j) + 1
                z = torch.cat((
                    features_l[:, i, torch.tensor([j] * d_range)].view(-1, self.fmaps),
                    features_r[:, i, j - torch.arange(d_range)].view(-1, self.fmaps)
                ), 1)
                outputs[i, j] = F.pad(self.fcnet(z).view(-1), (0, d_max + 1 - d_range)).detach()
        
        return outputs

    def __preprocess(self, data_path):
        self.train_imgs_l = []
        self.train_imgs_r = []
        self.disps = []
        self.train_pairs = []
        self.train_labels = []

        data_path = Path(data_path)
        for idx, datum_path in enumerate(list(data_path.iterdir())[:]):
        # for idx in range(10):
            print(datum_path)
            if datum_path.is_dir():
            # if True:
                img_l_path = datum_path / 'im0.png'
                img_r_path = datum_path / 'im1.png'
                disp_path = datum_path / 'disp0.pfm'
                # img_l_path = data_path / 'TL{}.png'.format(idx)
                # img_r_path = data_path / 'TR{}.png'.format(idx)
                # disp_path = data_path / 'TLD{}.pfm'.format(idx)

                img_l = cv2.imread(str(img_l_path))
                img_r = cv2.imread(str(img_r_path))
                disp = readPFM(str(disp_path))

                img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
                img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

                img_l = (img_l - np.mean(img_l)) / np.std(img_l)
                img_r = (img_r - np.mean(img_r)) / np.std(img_r)

                self.train_imgs_l.append(img_l)
                self.train_imgs_r.append(img_r)
                self.disps.append(disp)

                h, w = img_l.shape
                half_patch_size = self.patch_size // 2

                for i in range((h - self.patch_size + 1) // 4, (h - self.patch_size + 1) // 4 * 3): # DEBUG
                    for j in range((w - self.patch_size + 1) // 4, (w - self.patch_size + 1) // 4 * 3):
                        d = disp[i + half_patch_size, j + half_patch_size]

                        neg_delta = self.data_neg_high - self.data_neg_low

                        if j - d <= -self.data_pos:
                            continue

                        j_neg = None
                        while True:
                            o_neg = np.random.uniform(-neg_delta, neg_delta)
                            if (o_neg < 0):
                                o_neg -= self.data_neg_low
                            else:
                                o_neg += self.data_neg_low
                            j_neg = np.round(j - d + o_neg).astype(np.int)
                            if 0 <= j_neg and j_neg < w - self.patch_size + 1:
                                break
                        
                        j_pos = None
                        while True:
                            o_pos = np.random.uniform(-self.data_pos, self.data_pos)
                            j_pos = np.round(j - d + o_pos).astype(np.int)
                            if 0 <= j_pos and j_pos < w - self.patch_size + 1:
                                break
                        
                        pair_neg = np.array([[i, j], [i, j_neg], [idx, -1]])
                        pair_pos = np.array([[i, j], [i, j_pos], [idx, -1]])

                        self.train_pairs.append(pair_neg)
                        self.train_labels.append([0])

                        self.train_pairs.append(pair_pos)
                        self.train_labels.append([1])

        self.train_imgs_l = np.array(self.train_imgs_l)
        self.train_imgs_r = np.array(self.train_imgs_r)
        self.disps = np.array(self.disps)
        self.train_pairs = np.array(self.train_pairs)
        self.train_labels = np.array(self.train_labels)

        # self.train_imgs_l = torch.from_numpy(self.train_imgs_l).float().to(self.device)
        # self.train_imgs_r = torch.from_numpy(self.train_imgs_r).float().to(self.device)
        self.train_pairs = torch.from_numpy(self.train_pairs)
        self.train_labels = torch.from_numpy(self.train_labels).float()

