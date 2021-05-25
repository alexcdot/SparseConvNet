import torch

data1 = [[[0.88, 0.44,    0, 0.16, 0.37,    0, 0.96, 0.27],
          [   0, 0.45, 0.57, 0.16, 0.63, 0.29,    0,    0],
          [0.66,    0, 0.82, 0.64, 0.54,    0, 0.59, 0.26],
          [0.85, 0.34, 0.76, 0.84, 0.29, 0.75, 0.62, 0.25],
          [0.32, 0.74, 0.21, 0.39, 0.34, 0.03, 0.33, 0.48],
          [0.20, 0.14, 0.16, 0.13, 0.73, 0.65, 0.96, 0.32],
          [0.19, 0.69, 0.09, 0.86, 0.88, 0.07, 0.01, 0.48],
          [0.83, 0.24, 0.97, 0.04, 0.24, 0.35, 0.50, 0.91]]]
in_size1 = len(data1[0])

roi_box1_1 = torch.LongTensor([[0, 0, 3, 6, 7]])
out_size1_1 = 3
ans1_1 = torch.tensor([[[[0.85, 0.84, 0.75],
                         [0.69, 0.88, 0.96],
                         [0.97, 0.24, 0.50]]]])

roi_box1_2 = torch.LongTensor([[0, 0, 0, 2, 2]])
out_size1_2 = 2
ans1_2 = torch.tensor([[[[0.88, 0.57],
                         [0.66, 0.82]]]])

roi_box1_3 = torch.LongTensor([[0, 5, 0, 7, 2]])
out_size1_3 = 3
ans1_3 = torch.tensor([[[[   0, 0.96, 0.27],
                         [0.29,    0,    0],
                         [   0, 0.59, 0.26]]]])

def convert_data_to_sparse(data):
    # data must have shape (1, size, size) whose elements are numbers
    locations, features = [], []
    for y, line in enumerate(data[0]):
        for x, val in enumerate(line):
            if val != 0:
                locations.append([y, x, 0])
                features.append([val])
    locations = torch.LongTensor(locations)
    features = torch.FloatTensor(features)
    return locations, features
