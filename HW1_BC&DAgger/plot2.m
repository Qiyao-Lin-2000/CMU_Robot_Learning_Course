% -- Data you provided --------------------------------

% Ant-v2 DAgger (DAgger average returns = 9 data points)
ant_dagger_avg = [4712, 4792, 4369, 4520, 4624, 4682, 4702, 4730, 4722, 4751];
% (DAgger std returns = 10 data points)
ant_dagger_std = [72.92, 86.97, 843.5, 492.8, 328.9, 212, 168.8, 134.1, 118, 114.5];

% HalfCheetah-v2 DAgger (10 data points each)
cheetah_dagger_avg = [4024, 4098, 4088, 4094, 4107, 4117, 4125, 4102, 4100, 4146];
cheetah_dagger_std = [60.77, 77.65, 68, 68.71, 82.69, 78.18, 67.91, 82.29, 67.96, 67.96];

% Expert & BC "horizontal lines" for Ant-v2
ant_expert_avg = 4713;  % expert AverageReturn
ant_expert_std = 12.20; 
ant_bc_avg     = 4712;  
ant_bc_std     = 72.92;

% Expert & BC "horizontal lines" for HalfCheetah-v2
cheetah_expert_avg = 4205.78;
cheetah_expert_std = 83.04;
cheetah_bc_avg     = 4024.04;
cheetah_bc_std     = 60.77;

% -- Define x-axes (DAgger iteration indices) ----------
ant_iter_avg = 1:length(ant_dagger_avg);  % 1..9
ant_iter_std = 1:length(ant_dagger_std);  % 1..10

cheetah_iter_avg = 1:length(cheetah_dagger_avg); 
cheetah_iter_std = 1:length(cheetah_dagger_std);

% -- Create the figure with 2x2 subplots --------------
figure('Name','DAgger Results','NumberTitle','off');

% ============ Top-Left: Ant-v2 AverageReturn ============
subplot(2,2,1);
plot(ant_iter_avg, ant_dagger_avg, 'o-','LineWidth',1.2); 
hold on;
yline(ant_expert_avg, '--g', 'Expert');
yline(ant_bc_avg,     '--r', 'BC');
title('Ant-v2: DAgger AverageReturn');
xlabel('DAgger Iteration');
ylabel('Return');
grid on;
legend('DAgger','Expert','BC','Location','best');

% ============ Bottom-Left: Ant-v2 StdReturn ============
subplot(2,2,3);
plot(ant_iter_std, ant_dagger_std, 'o-','LineWidth',1.2);
hold on;
yline(ant_expert_std, '--g', 'Expert');
yline(ant_bc_std,     '--r', 'BC');
title('Ant-v2: DAgger StdReturn');
xlabel('DAgger Iteration');
ylabel('Std of Return');
grid on;
legend('DAgger','Expert','BC','Location','best');

% ============ Top-Right: HalfCheetah-v2 AverageReturn ============
subplot(2,2,2);
plot(cheetah_iter_avg, cheetah_dagger_avg, 'o-','LineWidth',1.2);
hold on;
yline(cheetah_expert_avg, '--g', 'Expert');
yline(cheetah_bc_avg,     '--r', 'BC');
title('HalfCheetah-v2: DAgger AverageReturn');
xlabel('DAgger Iteration');
ylabel('Return');
grid on;
legend('DAgger','Expert','BC','Location','best');

% ============ Bottom-Right: HalfCheetah-v2 StdReturn ============
subplot(2,2,4);
plot(cheetah_iter_std, cheetah_dagger_std, 'o-','LineWidth',1.2);
hold on;
yline(cheetah_expert_std, '--g', 'Expert');
yline(cheetah_bc_std,     '--r', 'BC');
title('HalfCheetah-v2: DAgger StdReturn');
xlabel('DAgger Iteration');
ylabel('Std of Return');
grid on;
legend('DAgger','Expert','BC','Location','best');

% Adjust spacing
set(gcf, 'Position', [100 100 1000 600]);
