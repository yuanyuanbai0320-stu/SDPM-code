clc 
clear
close all

pinghua = readmatrix('D:\采集数据\11221工作面\pinghua.xlsx');
% ====== 第一步：载入数据并选取第1列 ======
load yuanshi
load rnn
load lstm
load arma
load rsit

rmse2 = sqrt(mean((pinghua- arma).^2));
rmse4 = sqrt(mean((pinghua - rnn).^2));
rmse3 = sqrt(mean((pinghua - lstm).^2));
rmse1 = sqrt(mean((pinghua - rsit).^2));
% 计算 MAE
mae2 = mean(abs(pinghua - arma));
mae4 = mean(abs(pinghua - rnn));
mae3 = mean(abs(pinghua - lstm));
mae1 = mean(abs(pinghua - rsit));