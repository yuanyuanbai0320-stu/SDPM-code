function [all_weights, reverse_all_weights]=weights_align1(weight_a,weight_b,weight_c,input_dimension)
post_nodes_num=[weight_a,weight_b,weight_c];    %三个隐层的节点数量
layer_nodes_num=[input_dimension, post_nodes_num];
all_weights={};
reverse_all_weights={};
for l=2:length(layer_nodes_num)
    all_weights{l-1}=randn(layer_nodes_num(l-1),layer_nodes_num(l));%包含每个隐藏层的权重矩阵的单元格数组
    reverse_all_weights{length(layer_nodes_num)-l+1}=pinv(all_weights{l-1});
    %包含每个隐藏层的权重矩阵的逆矩阵的单元格数组
end