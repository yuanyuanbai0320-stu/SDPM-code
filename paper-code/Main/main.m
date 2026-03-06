clc
clear
close all

% -------------------- 载入数据 --------------------
load rmse_station_all
rmse_station_all_use = rmse_station_all;

load combined_sd_y
combined_sd_y_sc1 = combined_sd_y;

% 预处理ydata，补零对齐
ydata = [zeros(1,366-length(combined_sd_y_sc1)) combined_sd_y_sc1];

% 预处理rmse_stationall，补零对齐
rmse_station_all_predi_4 = [zeros(12, 366 - size(rmse_station_all_use, 2)) rmse_station_all_use];
rmse_stationall = rmse_station_all_predi_4;

% -------------------- 计算 p 值 --------------------
windows = 12;
for j = 1:size(rmse_stationall, 1)
    now_rmse_station = rmse_stationall(j, :);
    for i = windows+1:length(now_rmse_station)
        long_pic = now_rmse_station(i-windows:i-1);
        now_num = now_rmse_station(i);
        [h, p] = ttest(long_pic, now_num);
        loss_p(j, i) = p;
    end
end

% -------------------- 计算YYY，用于判断ydata的异常 --------------------
YYY = zeros(1, length(ydata));
for i = 1:length(ydata)
    YY_pic = ydata(11:14);
    YY_num = ydata(i);
    YYY(i) = ttest(YY_pic, YY_num, 'Alpha', 0.01);
end

% 筛选出YYY>0的点索引
k = 1;
clear mark1
for i = 1:length(YYY)
    if YYY(i) > 0
        mark1(k) = i;
        k = k + 1;
    end
end

% -------------------- 处理地震矩阵 --------------------
load magnitude
earthquake_matrix = magnitude(1, :);

for i = 1:length(earthquake_matrix)
    if earthquake_matrix(i) < 4.30
        earthquake_matrix(i) = 0;
    end
end
Y2 = earthquake_matrix;

% 时间轴
t_end = datetime(2023, 11, 10);
t_long = t_end - length(ydata) + 1;
t1 = t_long + caldays(0:length(ydata)-1);

% -------------------- 处理loss_p平均 --------------------
number = size(rmse_stationall, 1);
mmm = sum(loss_p(1:number, :));
mmm = mmm ./ number;
for i = 1:length(mmm)
    if mmm(i) ~= 0
        mmm(i) = 1 / mmm(i);
    end
end

% 找到mmm中大于20的点
k = 1;
for i = 1:length(mmm)
    if mmm(i) > 20
        mark3(k) = i;
        k = k + 1;
    end
end

% -------------------- 综合标记 --------------------
k = 1;
clear mark
earthquake_all = [];
accur_earthqu = [];
for i = 1:length(mark1)
    m = mark1(i);
    a = max(1, m - 14);
    b = min(length(ydata), m);
    c = min(length(ydata), m + 14);
    for j = 1:length(mark3)
        if mark3(j) >= a && mark3(j) <= b
            mark(k) = m;
            k = k + 1;
            break
        elseif mark3(j) > b && mark3(j) <= c
            mark(k) = mark3(j);
            k = k + 1;
            break
        end
    end
end
mark = unique(mark);

single = zeros(1, length(ydata));
for i = 1:length(mark)
    a = mark(i);
    single(a) = 1;
end

% -------------------- 正确预测和统计指标 --------------------
starday = 1;
earthquake = 0;
k = 1;
acuur_single = zeros(1, length(single));
for i = starday:length(Y2)
    if Y2(i) ~= 0 % 真实地震
        earthquake = earthquake + 1; % 地震计数
        a = max(1, i - 14);
        b = min(length(ydata), i);
        for j = a:b
            if single(j) == 1
                accur_earthqu(k) = i; % 正确预警的地震
                acuur_single(j) = 1;
                k = k + 1;
            end
        end
    end
end
accur_earthqu = unique(accur_earthqu);

k = 1;
acuur_mark = [];
for uu = 1:length(acuur_single)
    if acuur_single(uu) == 1
        acuur_mark(k) = uu;
        k = k + 1;
    end
end

TP = length(accur_earthqu);
FN = earthquake - TP;

% 剔除正确预测的点，剩余为误报
erro_single = single;
for i = 1:length(accur_earthqu)
    mmm = accur_earthqu(i);
    a = max(1, mmm - 14);
    b = min(length(ydata), mmm);
    for j = a:b
        if single(j) == 1
            erro_single(j) = 0; % 标记为正确预测，排除误报
        end
    end
end

k = 1;
erro_mark = [];
for uu = 1:length(erro_single)
    if erro_single(uu) == 1
        erro_mark(k) = uu;
        k = k + 1;
    end
end

FP = sum(erro_single(1, starday:end));
TN = length(ydata(starday:end)) - earthquake - FP;

Accuracy = (TP + TN) / (TP + TN + FP + FN);
Precision = TP / (TP + FP);
Recall = TP / (TP + FN);
FScore = (2 * Precision * Recall) / (Precision + Recall);
FPR = FP / (FP + TN);
FNR = FN / (TP + FN);

% 把预测成功和误报点的单点标记设为4，用于后面绘图区分
for i = 1:length(acuur_single)
    if acuur_single(i) ~= 0
        acuur_single(i) = 4;
    end
end
for i = 1:length(erro_single)
    if erro_single(i) ~= 0
        erro_single(i) = 4;
    end
end

% -------------------- 绘图部分 --------------------

% 日期范围
dates = datetime(2022, 11, 10):datetime(2023, 11, 10);
datenum_dates = datenum(dates);

% 设置背景颜色
bgColor = [1 1 1];

figure('Color', bgColor);
hold on
ax = gca;
ax.Color = bgColor;

% ================= 蓝色柱状（震级减4后×2） =================
Y_scaled = zeros(size(Y2));

for i = 1:length(Y2)
    if Y2(i) >= 4.27
        Y_scaled(i) = (Y2(i) - 4) * 2;   % 减4后再×2
    end
end

bar(datenum_dates, -Y_scaled, ...
    'FaceColor', [0.20, 0.45, 0.70], ...
    'EdgeColor', 'none', ...
    'BarWidth', 3);

% ================= 基线 =================
plot([datenum_dates(1), datenum_dates(end)], [0 0], ...
     'k-', 'LineWidth', 1);

% ================= 上半区 0–1 =================
w_num = 3;
y_base = 0;
h = 0.8;      % 高度改为1

% 正确预测（红）
for i = 1:length(acuur_mark)
    x_num = datenum_dates(acuur_mark(i));
    rectangle('Position', [x_num - w_num/2, y_base, w_num, h], ...
        'FaceColor', [0.85, 0.33, 0.31], ...
        'EdgeColor', 'none');
end

% 误报（灰）
for i = 1:length(erro_mark)
    x_num = datenum_dates(erro_mark(i));
    rectangle('Position', [x_num - w_num/2, y_base, w_num, h], ...
        'FaceColor', [0.6, 0.6, 0.6], ...
        'EdgeColor', 'none');
end

% 漏报（黄）
for i = 1:length(Y2)
    if Y2(i) >= 4.27
        a = max(1, i - 12);
        window = acuur_single(a:i);
        if all(window == 0)
            x_num = datenum_dates(i);
            rectangle('Position', [x_num - w_num/2, y_base, w_num, h], ...
                'FaceColor', [0.65, 0.55, 0.25], ...
                'EdgeColor', 'none');
        end
    end
end

% ================= 坐标轴设置 =================
xlim([datenum_dates(1), datenum_dates(end)]);
ylim([-1.1, 1.1]);

xticks(linspace(datenum_dates(1), datenum_dates(end), 6));
set(gca, 'XTickLabel', []);

set(gca, ...
    'YTick', [-1 0 1], ...
    'YTickLabel', [], ...
    'TickDir', 'out', ...
    'LineWidth', 3, ...
    'TickLength', [0.006 0.006], ...
    'FontName', 'Times New Roman', ...
    'FontSize', 18);

box off
set(gca, 'Layer', 'top')

hold off

set(gcf, 'Position', [100, 100, 1200, 380]);