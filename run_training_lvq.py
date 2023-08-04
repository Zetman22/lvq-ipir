import torch

import prototorch as pt

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


def load_raw_data_train(train, length):

  data, label = train[0].copy(), train[1].copy()
  #relabel the data
  for i in range(0,len(label)):
      if (label[i] == 'Id0')|(label[i] == 'id0'):
          label[i] = int(0)
      else:
          label[i] = int(1)
  k=1
  while (k<len(label)):
      if (label[k]==label[k-1]):
          label = np.delete(label,k)
          data[k-1].extend(data[k])
          data.pop(k)
      else:
          k=k+1
  #split the data in each data stamp. SHOULD CONCAT ALL STAMP
  trainls = []
  labells = []
  for i in range(len(data)):
      start = 0
      end = length - 1
      while end < len(data[i]):
          trainls.append(data[i][start:end+1])
          labells.append(label[i])
          start = start+length//4
          end = end+length//4
  x, y = trainls.copy(), labells.copy()
  del trainls, labells,
  return x, y

def load_raw_data_test(test, length):

  data_test, label_test = test[0].copy(), test[1].copy()
  #relabel the data
  for i in range(0,len(label_test)):
      if (label_test[i] == 'Rest')|(label_test[i] == 'id0'):
          label_test[i] = 0
      else:
          label_test[i] = 1
  k=1
  while (k<len(label_test)):
      if (label_test[k]==label_test[k-1]):
          label_test = np.delete(label_test,k)
          data_test[k-1].extend(data_test[k])
          data_test.pop(k)
      else:
          k=k+1
  #split the testset
  trainls_test = []
  labells_test = []
  for i in range(len(data_test)):
      start = 0
      end = length - 1
      while end < len(data_test[i]):
          trainls_test.append(data_test[i][start:end+1])
          labells_test.append(label_test[i])
          start = start+length//4
          end = end+length//4
  return trainls_test, labells_test


trainls_200, labells_200 = load_raw_data_train([data, label], length=200)
trainls_test_200, labells_test_200 = load_raw_data_test([data_test, label_test], length=200)
print(f"2.0s data train/test: {len(trainls_200)}, {len(trainls_test_200)}")


def extract_features(x_ls, y_ls, x_ls_test, y_ls_test):
  trainls, labells = x_ls.copy(), y_ls.copy()
  x = np.zeros((len(trainls),5))
  for i in range(len(trainls)):
      x[i][0] = min(trainls[i])
      index1 = np.argmin(trainls[i])
      x[i][1] = max(trainls[i])
      index2 = np.argmax(trainls[i])
      x[i][2] = np.mean(trainls[i])
      x[i][3] = np.std(trainls[i])
      x[i][4] = 100*(x[i][1]-x[i][0])/(index2-index1)
  y = labells

  trainls_test, labells_test = x_ls_test.copy(), y_ls_test.copy()
  x_test = np.zeros((len(trainls_test),5))
  for i in range(len(trainls_test)):
      x_test[i][0] = min(trainls_test[i])
      index1 = np.argmin(trainls_test[i])
      x_test[i][1] = max(trainls_test[i])
      index2 = np.argmax(trainls_test[i])
      x_test[i][2] = np.mean(trainls_test[i])
      x_test[i][3] = np.std(trainls_test[i])
      x_test[i][4] = 100*(x_test[i][1]-x_test[i][0])/(index2-index1)
  y_test = labells_test

  #Normalize data
  for i in range(len(x)):
    x[i] = (x[i] - np.mean(x[i]))/np.std(x[i])
  for v in range(len(x_test)):
    x_test[v] = (x_test[v] - np.mean(x_test[v]))/np.std(x_test[v])

  return x, y, x_test, y_test


x_200, y_200, x_test_200, y_test_200 = extract_features(trainls_200, labells_200, trainls_test_200, labells_test_200)
x = np.concatenate((x_200, x_test_200), axis=0)
y = np.concatenate((y_200, y_test_200), axis=0)


class DataWrapper(Dataset):
    def __init__(self, data):
        self.data = data  # list [x, y]

    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, idx):
        data_idx = idx % len(self.data[0])
        x = torch.tensor(self.data[0][data_idx])
        y = torch.tensor(self.data[1][data_idx])

        return x, y


def get_loader(args, data):
    
    def collate_fn(batch):
        x = torch.stack([item[0] for item in batch])
        y = torch.stack([item[1] for item in batch])
        return x, y
    
    dataset = DataWrapper(data)
    train_size = int(0.80 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                        generator=torch.Generator().manual_seed(5))

    train_loader =  DataLoader(dataset=train_dataset, 
                                drop_last=True,
                                shuffle=True,
                                collate_fn=collate_fn,
                                batch_size=args.batch_size,
                                num_workers=0)
    dev_loader =  DataLoader(dataset=val_dataset, 
                                drop_last=True,
                                shuffle=False,
                                collate_fn=collate_fn,
                                batch_size=args.batch_size,
                                num_workers=0)

    return train_loader, dev_loader


class GLVQ(torch.nn.Module):
    """
    Implementation of Generalized Learning Vector Quantization.
    """
    def __init__(self, data, **kwargs):
        super().__init__(**kwargs)

        self.components_layer = pt.components.LabeledComponents(
            distribution=[5, 5],  # number of codebooks each label
            components_initializer=pt.initializers.SMCI(data, noise=0.1),
        )

    def forward(self, data):
        components, label = self.components_layer()
      
        distance = pt.distances.squared_euclidean_distance(data, components)

        return distance, label

    def predict(self, data):
        """
        Predict the winning label from the distances to each codebook.
        """
        components, label = self.components_layer()
        distance = pt.distances.squared_euclidean_distance(data, components)
        winning_label = pt.competitions.wtac(distance, label)
        return winning_label


class GMLVQ(torch.nn.Module):
    """
    Implementation of Generalized Matrix Learning Vector Quantization.
    """
    def __init__(self, data, **kwargs):
        super().__init__(**kwargs)

        self.components_layer = pt.components.LabeledComponents(
            distribution=[5, 5],  # number of codebooks each label
            components_initializer=pt.initializers.SMCI(data, noise=0.1),
        )

        # Initialize Omega matrix
        self.backbone = pt.transforms.Omega(
            5,
            5,
            pt.initializers.RandomLinearTransformInitializer(),
        )

    def forward(self, data):
        components, label = self.components_layer()
        latent_x = self.backbone(data.unsqueeze(1) - components) ** 2 # (x - w) @ Omega.T
      
        distance = torch.sum(latent_x, dim=-1)

        return distance, label

    def predict(self, data):
        """
        Predict the winning label from the distances to each codebook.
        """
        components, label = self.components_layer()
        distance = torch.sum(self.backbone(data.unsqueeze(1) - components) ** 2, dim=-1)
        winning_label = pt.competitions.wtac(distance, label)
        return winning_label


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    train_loader, dev_loader = get_loader(args, [x, y])
    model = GMLVQ(train_loader).double()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = pt.losses.GLVQLoss(transfer_fn='identity')

    for epoch in range(1000):
        correct = 0.0
        for x, y in train_loader:
            d, labels = model(x)
            loss = criterion(d, y, labels).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                y_pred = model.predict(x)
                correct += (y_pred == y).float().sum(0)
        acc = 100 * correct / (len(train_loader) * args.batch_size)
        print(f"Epoch: {epoch} Accuracy: {acc:05.02f}%")

    correct_test = 0.0
    with torch.no_grad():
        for x, y in dev_loader:
            d, labels = model(x)
            y_pred = model.predict(x)
            correct_test += (y_pred == y).float().sum(0)
        acc_test = 100 * correct_test / (len(dev_loader) * args.batch_size)
        print(f"Accuracy test: {acc_test:05.02f}%")
