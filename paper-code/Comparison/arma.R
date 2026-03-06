#3.1修改ARMA
rm(list = ls())

library(readxl)
library(forecast)
library(openxlsx)
library(zoo)

# =====================================
# 1. 读取数据
# =====================================
GNSS_20220201 <- read_excel("D:/采集数据/11222工作面/11222_average.xlsx")
data_series_raw <- as.numeric(GNSS_20220201[[1]])

# =====================================
# 2. 滑动平均（必须和 MATLAB 一致）
# MATLAB: window_len = 4
# newdata(:, i)=mean(data(:, i: i+window_len-1))
# 等价于 align = "left"
# =====================================
window_len <- 4
data_series <- rollmean(data_series_raw, 
                        k = window_len, 
                        align = "left")

# =====================================
# 3. 参数设置（与 MATLAB 完全一致）
# =====================================
trainlength <- 9
predict_len <- 4
step_ahead <- predict_len - 1

n_total <- length(data_series)

all_predictions <- c()
all_true <- c()

ii <- 0

# =====================================
# 4. 严格等价 MATLAB 的 while 条件
# MATLAB: size(X,2)-ii >= trainlength + predict_len - 1
# =====================================
while ((n_total - ii) >= (trainlength + predict_len - 1)) {
  
  ii <- ii + 1
  
  train_data <- data_series[ii:(ii + trainlength - 1)]
  true_future <- data_series[(ii + trainlength):(ii + trainlength + step_ahead - 1)]
  
  # 自动选择 ARMA(p,q)
  model <- auto.arima(train_data, seasonal = FALSE)
  
  forecast_result <- forecast(model, h = step_ahead)
  pred_future <- as.numeric(forecast_result$mean)
  
  all_predictions <- c(all_predictions, pred_future)
  all_true <- c(all_true, true_future)
}

# =====================================
# 5. 计算误差
# =====================================
rmse_arma <- sqrt(mean((all_predictions - all_true)^2))
mae_arma  <- mean(abs(all_predictions - all_true))
mape_arma <- mean(abs((all_predictions - all_true) / all_true)) * 100

cat("====================================\n")
cat("Sliding Window ARMA Results\n")
cat("Total Prediction Points =", length(all_predictions), "\n")
cat("RMSE =", rmse_arma, "\n")
cat("MAE  =", mae_arma, "\n")
cat("MAPE =", mape_arma, "%\n")
cat("====================================\n")

# =====================================
# 6. 导出 Excel
# =====================================
comparison_df <- data.frame(
  Index = 1:length(all_true),
  Actual = all_true,
  Predicted = all_predictions,
  Error = all_predictions - all_true
)

write.xlsx(
  comparison_df,
  file = "D:/采集数据/11222工作面/ARMA_Sliding_Prediction_Result(zhen).xlsx",
  overwrite = TRUE
)

print(head(comparison_df))