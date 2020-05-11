% @author: Ning Wang

function phase_field_oneD()

clc; close all;

plt = 1;
plt_pred = 0;
save_plt = 1;

addpath Utilities
addpath Kernels/Phase_Field_oneD
addpath Utilities/export_fig

    function CleanupFun()
        rmpath ..
        rmpath Utilities
        rmpath Kernels/Phase_Field_oneD
        rmpath Utilities/export_fig
    end

finishup = onCleanup(@() CleanupFun());

set(0,'defaulttextinterpreter','latex')

%% Load Data
%load('../data/phase_field_oneD_simulation_beginning_stage_0.02noise.mat', 'usol', 't', 'x')
load('../data/phase_field_oneD_experiment.mat', 'usol', 't', 'x')
u_star = real(usol); % 500x500
t_star = t; % 500x1
x_star = x';   % 500x1
N_star = size(x_star,1);
nsteps = size(t_star,1)-1;

%% Setup
N0 = 420;
N1 = 410;
%% Clean Data
rng('default')
i = randi(nsteps);
fprintf(1,'random step: %d\n\n', i);

dt = t_star(i+1) - t_star(i);

idx0 = randsample(N_star, N0);

x0 = x_star(idx0,:);
u0 = u_star(idx0,i);

idx1 = randsample(N_star,N1);
x1 = x_star(idx1,:);
u1 = u_star(idx1,(i+1));

hyp = [log([1.0 1.0]) 20. 5. -1. -4.0];
model = HPM(x1, u1, x0, u0, dt, hyp);
model = model.train(5000);

hyp = model.hyp;
params = hyp(3:5);

[pred_n_star, var_n_star] = model.predict(x_star);
var_n_star = abs(diag(var_n_star));

error = norm(pred_n_star - u_star(:,i+1))/norm(u_star(:,i+1));

fprintf(1,'=========================\n');
fprintf(1,'Step: %d, Time = %.2f\n\nNLML = %.2f, Error = %.2e\n\n', i, ...
    t_star(i+1), model.NLML, error);

str = sprintf('%.4f  ', params);
fprintf('Parameters: %s\n\n', str)
fprintf(1,'=========================\n\n');

if plt_pred == 1
    figure();
    plot_prediction_1D(x_star, u_star(:,i+1), pred_n_star, var_n_star, ...
        '$x$', '$u(t,x)$', 'Prediction (clean data)');
    
    drawnow;
end

end