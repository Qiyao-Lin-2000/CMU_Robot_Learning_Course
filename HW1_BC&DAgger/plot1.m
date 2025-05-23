% 数据定义
train_steps = [15000, 20000, 25000, 30000, 35000];
avg_returns = [3955, 3942, 3978, 4024, 4101];
std_returns = [133.37, 145, 68, 60, 71];

% 创建画布
figure('Position', [100, 100, 800, 600])

% 子图1：平均回报 vs 训练步数
subplot(2,1,1)
plot(train_steps, avg_returns, '-o', 'LineWidth', 1.5, 'MarkerSize', 8, 'Color', [0.2 0.5 0.8])
title('Evaluation Average Return vs Training Steps')
xlabel('Number of Training Steps per Iteration')
ylabel('Average Return')
grid on
ylim([3800, 4200])  % 根据数据范围调整
xticks(train_steps)

% 子图2：标准差 vs 训练步数
subplot(2,1,2)
plot(train_steps, std_returns, '-s', 'LineWidth', 1.5, 'MarkerSize', 8, 'Color', [0.8 0.3 0.2])
title('Evaluation Return Std vs Training Steps')
xlabel('Number of Training Steps per Iteration')
ylabel('Standard Deviation')
grid on
ylim([0, 160])  % 根据数据范围调整
xticks(train_steps)

% 统一横轴范围
linkaxes([subplot(2,1,1), subplot(2,1,2)], 'x')
xlim([14000, 36000])