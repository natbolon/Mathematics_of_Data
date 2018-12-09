%%%%%%%%%%%%%%%%%%%%
%%% Exercise 3.4 %%%
%%%%%%%%%%%%%%%%%%%%

%% Comparison FISTA and linear estimator

format short
clear all
addpath('utilities/');
load('IMAGES/brain.mat')


m           = 256; 
rate        = 0.20;
%p           = numel(ind);

N           = m^2;

f           = im_brain;

% Optimization parameters
maxit   = 100;
tolx    = 1e-5;
regularization_parameter_lasso = 0.0005; 
Lips    = 1;

%% Define operators

% Wavelet operators
wav         = daubcqf(8);
level       = log2(m); % Maximum level
% Adjoint wavelet transform - From wavelet coefficients to image
WT          = @(x) midwt(real(x),wav,level) + 1i*midwt(imag(x),wav,level)  ; 
% Wavelet transform -  From image to wavelet coefficient
W           = @(x) mdwt(real(x),wav,level) +  1i*mdwt(imag(x),wav,level)   ; 


% Measurement operators
measurement_forward             = @(x, indices) fft2fwd_without_fftshift(reshape(x,[m,m]),indices);      % P_Omega F
measurement_backward            = @(x, indices) reshape(fft2adj_without_fftshift(x,indices,m,m),[N,1]);  % F^T P_Omega ^T 

% Vectorized transformations
representation_operator         = @(x) reshape(W(reshape(x,[m,m])),[N,1]);
representation_operator_trans   = @(x) reshape(WT(reshape(x,[m,m])),[N,1]);

% Define the overall operator
forward_operator                = @(x,indices) measurement_forward(representation_operator_trans(x), indices); % A
adjoint_operator                = @(x,indices) representation_operator(measurement_backward(x, indices)); % A^T 

%% Define PSNR 

psnr        = @(I, I_trans) 20*log10(max(max(I))/sqrt((1/N)*norm(I - I_trans)^2));
    
%% Define Operators
fx          = @(x,b, ind, p) 0.5*norm( b(1:p) - real(forward_operator(x(1:N), ind)) + imag(forward_operator(x(N+1:end), ind)),2)^2 + ...
                   0.5*norm(b(p+1:end) - imag(forward_operator(x(1:N), ind)) - real(forward_operator(x(N+1:end), ind)),2)^2;

gradf       = @(x,b, ind) gradient_f_comp(b, forward_operator, adjoint_operator, x, ind);

gx          = @(x) regularization_parameter_lasso*norm(x(1:N) +1i*x(N+1:end),1);

proxg       = @(x, gamma) proxL1norm_complex(x, regularization_parameter_lasso*gamma);




%% Compute linear estimator for random indices

ind  = randperm(N);
p    = round(rate*N);
ind  = reshape(sort(ind(1:p)), [p,1]);

% Generate measurements on complex domain and transfer to real domain
b           = [real(measurement_forward(f, ind)); imag(measurement_forward(f, ind))];

% Initial point
x0          = [zeros(N,1); zeros(N,1)]; % Vector size [2*m*m]

% Generate Linear reconstruction
time_lin    = tic;
x_rec_lin   = reshape(measurement_backward(b(1:p) + 1i*b(p+1:end), ind), [m,m]);
time_lin    = toc(time_lin)

% Execute FISTA
time_wav    = tic;
[f_wav, ~]  = FISTA_comparison(fx, gx, gradf, proxg, x0, Lips, maxit, tolx, b, ind);
time_wav    = toc(time_wav)

% Apply adjoint wavelet transformation to the output of FISTA
F = representation_operator_trans(f_wav(1:N) + 1i*f_wav(N+1:end));
F = reshape(F, [m,m]);

% Compute PSNR 
psnr_linear = abs(psnr(f, x_rec_lin));
psnr_wav    = abs(psnr(f, F));

% Plot Result and save Image
fig         = figure;
fontsize    = 16;
imagesc(abs(x_rec_lin), [min(abs(x_rec_lin(:))), max(abs(x_rec_lin(:)))]),  axis image off;
title(strcat('Random Indices PSNR= ', num2str(psnr_linear)),'fontsize',fontsize,'interpreter','latex');
file_name   = 'Images-34/Random_indices_linear';
saveas(fig, file_name, 'epsc');

% Plot Result and save Image
fig         = figure;
fontsize    = 16;
imagesc(abs(x_rec_lin), [min(abs(x_rec_lin(:))), max(abs(x_rec_lin(:)))]),  axis image off;
title(strcat('FISTA Random Indices PSNR= ', num2str(psnr_wav)),'fontsize',fontsize,'interpreter','latex');
file_name   = 'Images-34/Random_indices_wav';
saveas(fig, file_name, 'epsc');

% Plot Mask
M = zeros([N,1]);
M(ind) = 1;
M = reshape(M, [m,m]);
imagesc(M)
title('Random Indices Mask','fontsize',fontsize,'interpreter','latex');
file_name   = 'Images-34/Random_indices_mask';
saveas(fig, file_name, 'epsc');

%% Compute linear estimator for best indices

ind    = find(reshape(best_inds(f,rate), [N,1]));
p           = size(ind);

% Generate measurements on complex domain and transfer to real domain
b           = [real(measurement_forward(f, ind)); imag(measurement_forward(f, ind))];

% Initial point
x0          = [zeros(N,1); zeros(N,1)]; % Vector size [2*m*m]

% Generate Linear reconstruction
time_lin    = tic;
x_rec_lin   = reshape(measurement_backward(b(1:p) + 1i*b(p+1:end), ind), [m,m]);
time_lin    = toc(time_lin)

% Execute FISTA
time_wav    = tic;
[f_wav, ~]  = FISTA_comparison(fx, gx, gradf, proxg, x0, Lips, maxit, tolx, b, ind);
time_wav    = toc(time_wav)

% Apply adjoint wavelet transformation to the output of FISTA
F = representation_operator_trans(f_wav(1:N) + 1i*f_wav(N+1:end));
F = reshape(F, [m,m]);

% Compute PSNR 
psnr_linear = abs(psnr(f, x_rec_lin));
psnr_wav    = abs(psnr(f, F));

% Plot Result and save Image
fig         = figure;
fontsize    = 16;
imagesc(abs(x_rec_lin), [min(abs(x_rec_lin(:))), max(abs(x_rec_lin(:)))]),  axis image off;
title(strcat('Best Indices PSNR= ', num2str(psnr_linear)),'fontsize',fontsize,'interpreter','latex');
file_name   = 'Images-34/Best_indices_linear';
saveas(fig, file_name, 'epsc');

% Plot Result and save Image
fig         = figure;
fontsize    = 16;
imagesc(abs(x_rec_lin), [min(abs(x_rec_lin(:))), max(abs(x_rec_lin(:)))]),  axis image off;
title(strcat('FISTA Best Indices PSNR= ', num2str(psnr_wav)),'fontsize',fontsize,'interpreter','latex');
file_name   = 'Images-34/Best_indices_wav';
saveas(fig, file_name, 'epsc');

% Plot Mask
M = zeros([N,1]);
M(ind) = 1;
M = reshape(M, [m,m]);
imagesc(M)
title('Best Indices Mask','fontsize',fontsize,'interpreter','latex');
file_name   = 'Images-34/Best_indices_mask';
saveas(fig, file_name, 'epsc');

%% Compute linear estimator for initially given mask
load('IMAGES/indices_brain.mat')
p = numel(ind);

% Generate measurements 
b           = [real(measurement_forward(f, ind)); imag(measurement_forward(f, ind))];

% Initial point
x0          = [zeros(N,1); zeros(N,1)]; % Vector size [2*m*m]

% Generate Linear reconstruction
time_lin    = tic;
x_rec_lin   = reshape(measurement_backward(b(1:p) + 1i*b(p+1:end), ind), [m,m]);
time_lin    = toc(time_lin)

% Execute FISTA
time_wav    = tic;
[f_wav, ~]  = FISTA_comparison(fx, gx, gradf, proxg, x0, Lips, maxit, tolx, b, ind);
time_wav    = toc(time_wav)

% Apply adjoint wavelet transformation to the output of FISTA
F = representation_operator_trans(f_wav(1:N) + 1i*f_wav(N+1:end));
F = reshape(F, [m,m]);

% Compute PSNR 
psnr_linear = abs(psnr(f, x_rec_lin));
psnr_wav    = abs(psnr(f, F));

% Plot Result and save Image
fig         = figure;
fontsize    = 16;
imagesc(abs(x_rec_lin), [min(abs(x_rec_lin(:))), max(abs(x_rec_lin(:)))]),  axis image off;
title(strcat('Inital Indices PSNR= ', num2str(psnr_linear)),'fontsize',fontsize,'interpreter','latex');
file_name   = 'Images-34/Initial_indices_linear';
saveas(fig, file_name, 'epsc');

% Plot Result and save Image
fig         = figure;
fontsize    = 16;
imagesc(abs(x_rec_lin), [min(abs(x_rec_lin(:))), max(abs(x_rec_lin(:)))]),  axis image off;
title(strcat('FISTA Inital Indices PSNR= ', num2str(psnr_wav)),'fontsize',fontsize,'interpreter','latex');
file_name   = 'Images-34/Initial_indices_wav';
saveas(fig, file_name, 'epsc');

% Plot Mask
M = zeros([N,1]);
M(ind) = 1;
M = reshape(M, [m,m]);
imagesc(M)
title('Initial Indices Mask','fontsize',fontsize,'interpreter','latex');
file_name   = 'Images-34/Initial_indices_mask';
saveas(fig, file_name, 'epsc');
