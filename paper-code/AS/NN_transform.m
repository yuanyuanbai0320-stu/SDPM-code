function output = NN_transform(traindata, weights)
[input_dimension, trainlength]=size(traindata);
clear output;
for i=1:trainlength
    pre_layer_nodes_num=input_dimension;
    pre_layer_nodes_value=traindata(:,i)';         % input
    for l=1:length(weights)
        curr_weights=weights{l};
        curr_layer_nodes_value=tanh(pre_layer_nodes_value*curr_weights/3);
        curr_layer_nodes_value=myLeakyRelu(pre_layer_nodes_value*curr_weights,0.01, 1);
        pre_layer_nodes_value=curr_layer_nodes_value;
    end
    output(:, i)=curr_layer_nodes_value;
end