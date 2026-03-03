%% EE490 Project 1 - Part 2: Gradient Descent (AdaDelta-style RMS step)
clc; close all; clear;

%% ===== Define f(x,y) and gradient g(x,y) =====
f  = @(x,y) x.^2 + 2*x.*y + 4*y.^2;
gx = @(x,y) 2*x + 2*y;     % df/dx
gy = @(x,y) 2*x + 8*y;     % df/dy

%% ===== Plot f(x,y) surface in the required range [-2,2] =====
[xg, yg] = meshgrid(-2:0.05:2, -2:0.05:2);
fg = f(xg, yg);
figure; mesh(xg, yg, fg);
xlabel('x'); ylabel('y'); zlabel('f(x,y)');
title('Surface: f(x,y) = x^2 + 2xy + 4y^2');
hold on;

%% ===== AdaDelta-style parameters (from instruction) =====
eta   = 0.1;       % given initial learning rate
rho   = 0.95;      % choose in [0.9, 0.99]
eps0  = 1e-8;      % small constant for stability
tol_f = 1e-3;      % stop: |f_{t+1} - f_t| < 1e-3
maxIter = 5000;

%% ===== Initialization =====
x0 = 1.5; 
y0 = 1.5;

wt = [x0; y0];
Track = wt;

Eg2 = [0;0];        % running RMS accumulator for gradient squares
eta_eff_hist = [];  % store effective learning rate eta'

%% ===== Gradient Descent loop =====
t = 1;
f_old = f(wt(1), wt(2));

while t < maxIter
    
    % gradient at current point
    g = [gx(wt(1), wt(2));
         gy(wt(1), wt(2))];
    
    % running average of squared gradients
    Eg2 = rho*Eg2 + (1-rho)*(g.^2);
    
    % RMS of gradient
    RMSg = sqrt(Eg2 + eps0);
    
    % AdaDelta-style step (matches your handout formula)
    dx = -(eta ./ RMSg) .* g;
    
    % effective learning rate (scalar for plotting)
    eta_eff = mean(eta ./ RMSg);
    eta_eff_hist = [eta_eff_hist; eta_eff]; %#ok<AGROW>
    
    % update
    wt1 = wt + dx;
    
    % enforce bounds [-2,2]
    wt1(1) = max(-2, min(2, wt1(1)));
    wt1(2) = max(-2, min(2, wt1(2)));
    
    % compute new f
    f_new = f(wt1(1), wt1(2));
    
    % save path
    Track = [Track, wt1]; %#ok<AGROW>
    
    % stopping criterion
    if abs(f_new - f_old) < tol_f
        wt = wt1;
        break;
    end
    
    % move forward
    wt = wt1;
    f_old = f_new;
    t = t + 1;
end

xf = wt(1); yf = wt(2);

fprintf('Stopped at iteration  : %d\n', t);
fprintf('Final solution (xf,yf): (%.6f, %.6f)\n', xf, yf);
fprintf('Final f(xf,yf)        : %.8f\n', f(xf,yf));

%% ===== (a) Plot trajectory on the surface =====
xt = Track(1,:);
yt = Track(2,:);
ft = f(xt, yt);

plot3(xt, yt, ft, 'r*-', 'LineWidth', 1.4, 'MarkerSize', 5);
plot3(xf, yf, f(xf,yf), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 8);
title('AdaDelta-style Trajectory on f(x,y) Surface');
legend('Surface','Trajectory','Final point','Location','best');
view(45,35);
grid on;

%% ===== (b) Plot eta'' (effective learning rate) vs iteration =====
figure;
plot(1:length(eta_eff_hist), eta_eff_hist, 'b-', 'LineWidth', 1.6);
grid on;
xlabel('Iteration');
ylabel('\eta'' = mean(\eta ./ RMS[g])');
title('Effective Learning Rate \eta'''' vs Iteration (AdaDelta-style)');