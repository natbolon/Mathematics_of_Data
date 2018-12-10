%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Exercise 4.2             Convergence analysis %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Add paths
clear all
format long
addpath('utilities/');
addpath('IMAGES/')

%% Load Image and transform
m       = 1024;

F_star  = 3.035921667465773e+06; 


I       = imread('Randa.jpg'); 
I       = rgb2gray(I);
I       = double(I);
I       = imresize(I, [m, m]);

N       = m^2;
rate    = 0.4;

ind     = randperm(N);
p       = round(rate*N);
ind     = reshape(sort(ind(1:p)), [p,1]);

mask        = zeros([N,1]);
mask(ind)   = 1;
mask        = reshape(mask, [m,m]);

%% Define PSNR 
psnr        = @(I, I_trans) 20*log10(max(max(I))/sqrt((1/N)*norm(I - I_trans, 'fro')^2));


%% Wavelet operators
% Define the function handles that compute
% the products by W (DWT) and W' (inverse DWT)

wav         = daubcqf(8);
level       = log2(m); % Maximum level

% Adjoint wavelet transform - From wavelet coefficients to image
WT          = @(x) midwt(x,wav,level); 
% Wavelet transform -  From image to wavelet coefficient
W           = @(x) mdwt(x,wav,level); 

% Select Indices
S           = @(x) mask.*x;

b           = S(I);

%% Define Parameters 

maxit       = 2000;
tolx        = 1e-15;
Lips        = 1;
reg_lasso   = 10; 


%% Define Operators Norm 1
fx          = @(x) 0.5*norm(b - S(WT(x)),2)^2;

gradf       = @(x) -W(b - S(WT(x)));

gx          = @(x) reg_lasso*norm(x,1);

proxg       = @(x, gamma) proxL1norm(x, gamma*reg_lasso);


%% 
x0            = zeros(m);
fig = figure;

xlabel('Iterations')
ylabel('Relative error')
hold on

% Execute ISTA 
disp('Executing ISTA')
time    = tic;
[f_ISTA, error_ISTA]             = ISTA(fx, gx, gradf, proxg, x0, Lips, maxit, F_star);
disp(strcat('Time ISTA: ', num2str(toc(time))))

semilogy(error_ISTA, 'LineWidth',2)


% Execute FISTA without restart
disp('Executing FISTA without restart')
time    = tic;
[f_FISTA_nr, error_FISTA_nr]     = FISTA_no_restart(fx, gx, gradf, proxg, Lips, x0, maxit, F_star);
disp(strcat('Time: ', num2str(toc(time))))

semilogy(error_FISTA_nr, 'LineWidth',2)

% Execute FISTA fixed iteration restart
disp('Executing FISTA with restart every 20 iterations')
it_restart = 20;
time    = tic;
[f_FISTA_iter_20, error_FISTA_iter_20] = FISTA_iter_restart(fx, gx, gradf, proxg, x0, Lips, maxit, it_restart, F_star);
disp(strcat('Time: ', num2str(toc(time))))

semilogy(error_FISTA_iter_20, 'LineWidth',2)

disp('Executing FISTA with restart every 50 iterations')
it_restart = 50;
time    = tic;
[f_FISTA_iter_50, error_FISTA_iter_50] = FISTA_iter_restart(fx, gx, gradf, proxg, x0, Lips, maxit, it_restart, F_star);
disp(strcat('Time: ', num2str(toc(time))))

semilogy(error_FISTA_iter_50, 'LineWidth',2)

disp('Executing FISTA with restart every 100 iterations')
it_restart = 100;
time    = tic;
[f_FISTA_iter_100, error_FISTA_iter_100] = FISTA_iter_restart(fx, gx, gradf, proxg, x0, Lips, maxit, it_restart, F_star);
disp(strcat('Time: ', num2str(toc(time))))

semilogy(error_FISTA_iter_100, 'LineWidth',2)

disp('Executing FISTA with restart every 200 iterations')
it_restart = 200;
time    = tic;
[f_FISTA_iter_200, error_FISTA_iter_200] = FISTA_iter_restart(fx, gx, gradf, proxg, x0, Lips, maxit, it_restart, F_star);
disp(strcat('Time: ', num2str(toc(time))))

semilogy(error_FISTA_iter_200, 'LineWidth',2)

% Execute FISTA with gradeint scheme restart
disp('Executing FISTA with gradient scheme restart')
time    = tic;
[f_FISTA_grad, error_FISTA_grad] = FISTA_grad_restart(fx, gx, gradf, proxg, x0, Lips, maxit, F_star);
disp(strcat('Time: ', num2str(toc(time))))

semilogy(error_FISTA_grad, 'LineWidth',2)

legend('ISTA','FISTA no restart', 'FISTA 20 iter', 'FISTA 50 iter', 'FISTA 100 iter', 'FISTA 200 iter', 'FISTA gradient restart' )
%f_norm1        = WT(f_norm1);

% Reshape_images
%f_norm1        = reshape(f_norm1, [m,m]);


% Compute PSNR
%psnr_1          = psnr(I,f_norm1);

%%

figure
hold on
semilogy(error_FISTA_iter_20, 'LineWidth',2)
semilogy(error_FISTA_iter_50, 'LineWidth',2)
semilogy(error_FISTA_iter_100, 'LineWidth',2)
semilogy(error_FISTA_iter_200, 'LineWidth',2)
hold off

figure;
hold on
semilogy(error_ISTA, 'LineWidth',2)
semilogy(error_FISTA_nr, 'LineWidth',2)
semilogy(error_FISTA_grad, 'LineWidth',2)
hold off
