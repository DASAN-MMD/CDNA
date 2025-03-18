# https://www.kaggle.com/code/hirotaka0122/triplet-loss-with-pytorch
import torch


class SamplePairGenerator(Dataset):
    def __init__(self, df, train=True, transform=None):
        self.is_train = train
        self.transform = transform
        self.to_pil = transforms.ToPILImage()

        if self.is_train:
            self.images = df.iloc[:, 1:].values.astype(np.float32)
            self.labels = df.iloc[:, 0].values
            self.index = df.index.values
        else:
            self.images = df.values.astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        anchor_img = self.images[item].reshape(64, 64, 3)

        if self.is_train:
            anchor_label = self.labels[item]

            positive_list = self.index[self.index != item][self.labels[self.index != item] == anchor_label]

            positive_item = random.choice(positive_list)
            positive_img = self.images[positive_item].reshape(64, 64, 3)

            negative_list = self.index[self.index != item][self.labels[self.index != item] != anchor_label]
            negative_item = random.choice(negative_list)
            negative_img = self.images[negative_item].reshape(64, 64, 3)


            if self.transform:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)

            return anchor_img, positive_img, negative_img, anchor_label

        else:

            if self.transform:
                anchor_img = self.transform(anchor_img)
            return anchor_img


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor, positive, negative):
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


class AE(nn.Module):  # nn.Module is pre-defined class acting as a parent class here
    def __init__(self,input_size1,input_size2,hidden_size1,hidden_size2, num_layers):  # This is constructor is Python which will explicity being called once we create a instance of this class
    #def __init__(self):
        super(AE, self).__init__()  # This is inheritence in python for called parent init method defined in nn.Module class
        latent_size = 2000
        self.num_layers = num_layers
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        #self.unpool = nn.UpsamplingNearest2d(64)
        #self.dropout = nn.Dropout(0.3)
        self.rnn1 = nn.GRU(input_size1, hidden_size1, num_layers, batch_first=True)

        # Decoder

        self.rnn2 = nn.GRU(input_size2, hidden_size2, num_layers, batch_first=True)
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)
        self.fc1 = nn.Linear(64*100,latent_size)
        self.fc2 = nn.Linear(latent_size,64*100)
        self.latent = None

    def encode(self, x):
        h01 = torch.zeros(self.num_layers, x.size(0), self.hidden_size1).to(device)
        x = F.leaky_relu(self.conv1(x))
        x = self.pool(x)
        #print(x.shape)
        #x = self.dropout(x)
        #print(x.shape)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        #print(x.shape)
        x = x.view(-1, 64, 16)
        # print(x.shape)
        x, _ = self.rnn1(x, h01)
        #print(x.shape)
        x = x.reshape(-1, 6400)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        self.latent = x
        return x

    def decode(self, x):
        h02 = torch.zeros(self.num_layers, x.size(0), self.hidden_size2).to(device)
        x = F.relu(self.fc2(x))
        # print(x.shape)
        x = x.view(-1, 64, 100)
        x, _ = self.rnn2(x, h02)
        # print(x.shape)
        x = x.view(-1, 4, 16, 16)
        x = F.leaky_relu(self.t_conv1(x))
        #x = self.unpool(x)
        #print(x.shape)
        x = F.leaky_relu(self.t_conv2(x))
        #x = self.unpool(x)
        #print(x.shape)
        return x

    def forward(self, x):
        x = self.encode(x)
        # print(x.shape)
        x = self.decode(x)
        return x

class Aligner(nn.Module):
    def __init__(self):
        super(Aligner, self).__init__()
        # self.image_size = 14
        self.image_size = 64
        #self.batch_size = 1
        # read layer
        self.fc1 = nn.Linear(3*self.image_size * self.image_size, 3*self.image_size * self.image_size)

        # exp unit
        self.relu = nn.ReLU(.5)

        # out layer
        self.fc2 = nn.Linear(3*self.image_size * self.image_size, 3*self.image_size * self.image_size)

    def forward(self, x):
        self.batch_size = x.size(0)
        x = x.view(self.batch_size, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.view(self.batch_size, 3, self.image_size, self.image_size)
        return x

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.fc2 = nn.Linear(2000, 500)
        self.fc1 = nn.Linear(2000, 2000)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 7)
        self.relu = nn.ReLU(.5)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x1 = self.relu(x)
        x = self.fc4(x1)

        return F.log_softmax(x, dim=1), x1

def discrepancy(out1, out2):
    return torch.mean(torch.abs(out1 - out2))
