clc
clear
close all

%% ===============================
% 1. 载入数据
%% ===============================
load GNSS_20220201
data = GNSS_20220201;
data_single = data(:,1);

%% ===============================
% 2. 滑动平均（与前面方法一致）
%% ===============================
window_len = 4;
newdata = movmean(data_single,[0 window_len-1]);
newdata = newdata(1:end-window_len+1);

data_single = newdata;

%% ===============================
% 3. 参数设置（必须一致）
%% ===============================
trainlength = 9;
predict_len = 4;
step_ahead = predict_len - 1;

n_total = length(data_single);

all_predictions = [];
all_true = [];

ii = 0;

%% ===============================
% 4. 滑动窗口预测
%% ===============================
while (n_total - ii) >= (trainlength + predict_len - 1)
    
    ii = ii + 1;
    
    % 当前窗口数据
    train_data = data_single(ii : ii+trainlength-1);
    true_future = data_single(ii+trainlength : ...
        ii+trainlength+step_ahead-1);
    
    % ============================
    % 归一化（每个窗口单独归一化）
    % ============================
    mu = mean(train_data);
    sigma = std(train_data);
    trainNorm = (train_data - mu) / sigma;
    
    % ============================
    % 构造 LSTM 训练数据
    % ============================
    XTrain = trainNorm(1:end-1)';
    YTrain = trainNorm(2:end)';
    
    % ============================
    % LSTM 网络结构
    % ============================
    layers = [
        sequenceInputLayer(1)
        lstmLayer(30,'OutputMode','sequence')
        fullyConnectedLayer(1)
        regressionLayer];
    
    options = trainingOptions('adam', ...
        'MaxEpochs',100, ...
        'MiniBatchSize',1, ...
        'InitialLearnRate',0.01, ...
        'Verbose',0);
    
    % ============================
    % 训练
    % ============================
    net = trainNetwork(XTrain, YTrain, layers, options);
    
    % ============================
    % 多步递推预测
    % ============================
    YPredNorm = zeros(1,step_ahead);
    
    net = predictAndUpdateState(net, XTrain);
    [net, YPredNorm(1)] = predictAndUpdateState(net, YTrain(end));
    
    for k = 2:step_ahead
        [net, YPredNorm(k)] = predictAndUpdateState(net, YPredNorm(k-1));
    end
    
    % ============================
    % 反归一化
    % ============================
    YPred = YPredNorm * sigma + mu;
    
    % 拼接
    all_predictions = [all_predictions YPred];
    all_true = [all_true true_future'];
    
end

%% ===============================
% 5. 计算误差
%% ===============================
rmse_lstm = sqrt(mean((all_predictions - all_true).^2));
mae_lstm  = mean(abs(all_predictions - all_true));
mape_lstm = mean(abs((all_predictions - all_true)./all_true))*100;

disp('======================================')
disp(['Total Prediction Points = ', num2str(length(all_predictions))])
disp(['LSTM Sliding RMSE = ', num2str(rmse_lstm)])
disp(['LSTM Sliding MAE  = ', num2str(mae_lstm)])
disp(['LSTM Sliding MAPE = ', num2str(mape_lstm)])
disp('======================================')

%% ===============================
% 6. 保存结果
%% ===============================
ResultTable = table((1:length(all_true))', ...
    all_true', ...
    all_predictions', ...
    'VariableNames',{'Index','Actual','Predicted'});

writetable(ResultTable,'LSTM_Sliding_Result.xlsx');

%% ===============================
% 7. 可视化
%% ===============================
figure
plot(all_true,'b','LineWidth',1.5)
hold on
plot(all_predictions,'r--','LineWidth',1.5)
legend('True','LSTM Prediction')
title('Sliding Window LSTM Forecast')
grid on