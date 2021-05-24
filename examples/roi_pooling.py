import torch
import sparseconvnet as scn

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available() and scn.SCN.is_cuda_build()
    device = 'cuda:0' if use_cuda else 'cpu'

    out_size = 3
    # dimension = 2, pool_size = 3, pool_stride = 1, out_size = 3
    pooling_layer = scn.RoiPooling(2, out_size, 1, out_size).to(device)

    sparse_to_dense = scn.Sequential(
        scn.SparseToDense(2, 1)
    )

    # for roi of roi box[[3, 0, 7, 6]] with output shape [2, 2]
    # [[[0.85, 0.96], [0.86, 0.88]], [[0.88, 0.57], [0.85, 0.84]]]
    data_arr = [[[0.88, 0.44,    0, 0.16, 0.37, 0.77, 0.96, 0.27],
                 [   0, 0.45, 0.57, 0.16, 0.63, 0.29, 0.71, 0.70],
                 [0.66,    0, 0.82, 0.64, 0.54, 0.73, 0.59, 0.26],
                 [0.85, 0.34, 0.76, 0.84, 0.29, 0.75, 0.62, 0.25],
                 [0.32, 0.74, 0.21, 0.39, 0.34, 0.03, 0.33, 0.48],
                 [0.20, 0.14, 0.16, 0.13, 0.73, 0.65, 0.96, 0.32],
                 [0.19, 0.69, 0.09, 0.86, 0.88, 0.07, 0.01, 0.48],
                 [0.83, 0.24, 0.97, 0.04, 0.24, 0.35, 0.50, 0.91]]]
    roi_boxes = torch.LongTensor([[0, 0, 3, 6, 7]])

    # for roi of roi box[[0, 0, 2, 2]] with output shape [2, 2]
    # [[0.88, 0.57], [0.66, 0.82]]
    # data_arr = [[[0.88, 0.44,    0, 0.16, 0.37],
    #              [   0, 0.45, 0.57,    0, 0.63],
    #              [0.66,    0, 0.82, 0.64,    0],
    #              [0.85, 0.34, 0.76, 0.84,    0],
    #              [   0, 0.74,    0,    0, 0.34]]]
    # roi_boxes = torch.LongTensor([[0, 0, 0, 2, 2]])

    locations, features = [], []
    for batchIdx, data in enumerate(data_arr):
        for y, line in enumerate(data):
            for x, c in enumerate(line):
                if c == 'X':
                    locations.append([y, x, batchIdx])
                    features.append([1])
                elif c == ' ':
                    pass
                elif c != 0:
                    locations.append([y, x, batchIdx])
                    features.append([c])
    locations = torch.LongTensor(locations)
    features = torch.FloatTensor(features).to(device)

    # inputSpatialSize = pooling_layer.input_spatial_size(torch.LongTensor([2, 2]))
    # inputSpatialSize = pooling_layer.input_spatial_size(torch.LongTensor([3, 3]))
    inputSpatialSize = torch.tensor([5, 5])
    input_layer = scn.InputLayer(2, inputSpatialSize)
    input = input_layer([locations,features])
    output = pooling_layer(input, roi_boxes)
    locations = torch.LongTensor([[0 ,0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    
    # reorder features since the spatial location of output is not in row-order format
    out_locations = output.metadata.getSpatialLocations(torch.tensor([out_size, out_size]))
    out_updated_features = torch.zeros(out_size * out_size, 1)
    for idx, out_location in enumerate(out_locations):
        out_idx = out_location[0] * out_size + out_location[1]
        out_updated_features[idx][0] = output.features[out_idx][0]

    # reorder features
    output.features = out_updated_features

    output = sparse_to_dense(output)
    print(output)
