import torch
import sparseconvnet as scn

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available() and scn.SCN.is_cuda_build()
    device = 'cuda:0' if use_cuda else 'cpu'

    pooling_layer = scn.Sequential(
        scn.RoiPooling(2, 3, 2).to(device),
        # scn.MaxPooling(2, 4, 2).to(device),
    )

    sparse_to_dense = scn.Sequential(
        scn.SparseToDense(2, 1)
    )

    # for roi of [[3, 0, 7, 6]] with output shape [2, 2]
    # [[0.85, 0.84], [0.97, 0.96]]
    data_arr = [[[0.88, 0.44, 0.14, 0.16, 0.37, 0.77, 0.96, 0.27],
                 [0.19, 0.45, 0.57, 0.16, 0.63, 0.29, 0.71, 0.70],
                 [0.66, 0.26, 0.82, 0.64, 0.54, 0.73, 0.59, 0.26],
                 [0.85, 0.34, 0.76, 0.84, 0.29, 0.75, 0.62, 0.25],
                 [0.32, 0.74, 0.21, 0.39, 0.34, 0.03, 0.33, 0.48],
                 [0.20, 0.14, 0.16, 0.13, 0.73, 0.65, 0.96, 0.32],
                 [0.19, 0.69, 0.09, 0.86, 0.88, 0.07, 0.01, 0.48],
                 [0.83, 0.24, 0.97, 0.04, 0.24, 0.35, 0.50, 0.91]]]
    
    # expected output is [["10", "01"], ["11", "10"]]
    data_arr = [["X    ", "     ", "     ", "   X ", "     "],
                ["    X", "     ", "X    ", "     ", "     "]]
    # data_arr = [["X    ", "     ", "     ", "   X ", "     "]]
    # data_arr = [["    X", "     ", " X   ", "     ", "     "]]

    roi_boxes = [[0, 3, 0, 7, 6]]

    locations, features = [], []
    for batchIdx, data in enumerate(data_arr):
        for y, line in enumerate(data):
            for x, c in enumerate(line):
                if c == 'X':
                    locations.append([y, x, batchIdx])
                    features.append([1])
                elif c == ' ':
                    pass
                else:
                    locations.append([y, x, batchIdx])
                    features.append([c])
    print('locations', locations)
    print('features', features)
    locations = torch.LongTensor(locations)
    features = torch.FloatTensor(features).to(device)

    inputSpatialSize = pooling_layer.input_spatial_size(torch.LongTensor([2, 2]))
    # inputSpatialSize = pooling_layer.input_spatial_size(torch.LongTensor([3, 3]))
    input_layer = scn.InputLayer(2, inputSpatialSize)
    input = input_layer([locations,features])
    print(input.features)
    output = pooling_layer(input)
    print(output)
    output = sparse_to_dense(output)
    print(output)
