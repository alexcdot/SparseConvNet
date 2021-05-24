import torch
import sparseconvnet as scn

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available() and scn.SCN.is_cuda_build()
    device = 'cuda:0' if use_cuda else 'cpu'

    pooling_layer = scn.Sequential(
        # dimension = 2, pool size = 3, stride = 2
        scn.MaxPooling(2, 3, 2).to(device),
        scn.SparseToDense(2, 1)
    )

    # expected output is [["10", "01"], ["11", "10"]]
    data_arr = [["X    ", "     ", "     ", "     ", "   X ", "     "],
                ["    X", "     ", "X    ", "     ", "     ", "     "]]

    locations, features = [], []
    for batchIdx, data in enumerate(data_arr):
        for y, line in enumerate(data):
            for x, c in enumerate(line):
                if c == 'X':
                    locations.append([y, x, batchIdx])
                    features.append([1])
    locations = torch.LongTensor(locations)
    features = torch.FloatTensor(features).to(device)
    print(locations, features, sep='\n')
    inputSpatialSize = pooling_layer.input_spatial_size(torch.LongTensor([2, 2]))
    input_layer = scn.InputLayer(2, inputSpatialSize)
    input = input_layer([locations,features])
    output = pooling_layer(input)
    print(output)
