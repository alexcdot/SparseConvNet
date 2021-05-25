import sparseconvnet as scn
import torch

from cpu_demo_data import * 

def get_roi_pooling_output(in_data, in_size, roi_boxes, out_size):
    input_spatial_size = torch.tensor([in_size, in_size])
    input_layer = scn.InputLayer(2, input_spatial_size) # 2 is the dimension
    pooling_layer = scn.RoiPooling(2, out_size, 1, out_size)
    sparse_to_dense = scn.SparseToDense(2, 1)

    locations, features = convert_data_to_sparse(in_data)
    input_sparse = input_layer([locations, features])
    output_sparse = pooling_layer(input_sparse, roi_boxes)

    # re-order features b/c the spatial location of output is not in row-order format
    out_locations = output_sparse.metadata.getSpatialLocations(torch.tensor([out_size, out_size]))
    out_updated_features = torch.zeros(out_size * out_size, 1)
    for idx, out_location in enumerate(out_locations):
        out_idx = out_location[0] * out_size + out_location[1]
        out_updated_features[idx][0] = output_sparse.features[out_idx][0]
    output_sparse.features = out_updated_features

    output = sparse_to_dense(output_sparse)
    return output

def print_code_test(ans, output, test_name):
    print(f'********* {test_name} *********')
    print("Expected output: ")
    print(ans)
    print("Demo output: ")
    print(output)
    print("Is correct: ", torch.equal(ans, output))

if __name__ == '__main__':
    output1_1 = get_roi_pooling_output(data1, in_size1, roi_box1_1, out_size1_1)
    print_code_test(ans1_1, output1_1, "Test 1-1")

    output1_2 = get_roi_pooling_output(data1, in_size1, roi_box1_2, out_size1_2)
    print_code_test(ans1_2, output1_2, "Test 1-2")

    output1_3 = get_roi_pooling_output(data1, in_size1, roi_box1_3, out_size1_3)
    print_code_test(ans1_3, output1_3, "Test 1-3")