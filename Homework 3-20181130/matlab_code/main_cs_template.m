%% Problem size - image side
% addpath('export_fig');
clear all
format long
addpath('utilities/');
load('IMAGES/brain.mat')
load('IMAGES/indices_brain.mat')

m           = 256; 
rate        = 0.20;
p           = numel(ind);

N           = m^2;

f           = im_brain;

%% Wavelet operators
% Define the function handles that compute
% the products by W (DWT) and W' (inverse DWT)

wav         = daubcqf(8);
level       = log2(m); % Maximum level

% Adjoint wavelet transform - From wavelet coefficients to image
WT          = @(x) midwt(real(x),wav,level) + 1i*midwt(imag(x),wav,level)  ; 
% Wavelet transform -  From image to wavelet coefficient
W           = @(x) mdwt(real(x),wav,level) +  1i*mdwt(imag(x),wav,level)   ; 

 %% Define operators

% Vectorized transformations
representation_operator         = @(x) reshape(W(reshape(x,[m,m])),[N,1]);
representation_operator_trans   = @(x) reshape(WT(reshape(x,[m,m])),[N,1]);

% Measurement operators
measurement_forward             = @(x) fft2fwd_without_fftshift(reshape(x,[m,m]),ind);      % P_Omega F
measurement_backward            = @(x) reshape(fft2adj_without_fftshift(x,ind,m,m),[N,1]);  % F^T P_Omega ^T 


% Define the overall operator
forward_operator                = @(x) measurement_forward(representation_operator_trans(x)); % A
adjoint_operator                = @(x) representation_operator(measurement_backward(x)); % A^T 

%% Define PSNR 

psnr    = @(I, I_trans) 20*log10(max(max(I))/sqrt((1/N)*norm(I - I_trans)^2));
%%
Lips    = 1;

% Generate measurements on complex domain and transfer to real domain
b       = [real(measurement_forward(f)); imag(measurement_forward(f))];

% Optimization parameters
maxit = 100;
tolx  = 1e-5;

%% WAVELET RECOVERY

PSNR_list   = [];
regs        = logspace(-9,-0.1,15);
% Initial point
x0          = [zeros(N,1); zeros(N,1)]; % Vector size [2*m*m]

for k=1:length(regs)
    regularization_parameter_lasso = regs(k)
    % Define operators
    fx          = @(x) 0.5*norm( b(1:p) - real(forward_operator(x(1:N))) + imag(forward_operator(x(N+1:end))),2)^2 + ...
                   0.5*norm(b(p+1:end) - imag(forward_operator(x(1:N))) - real(forward_operator(x(N+1:end))),2)^2;

    gx          = @(x) regularization_parameter_lasso*norm(x(1:N) +1i*x(N+1:end),1);

    proxg       = @(x, gamma) proxL1norm_complex(x, regularization_parameter_lasso*gamma);

    gradf       = @(x) gradient_f(b, forward_operator, adjoint_operator, x);


    % Execute FISTA
    time_wav    = tic;
    [f_wav, ~]  = FISTA(fx, gx, gradf, proxg, x0, Lips, maxit, tolx);
    time_wav    = toc(time_wav)

    % Apply adjoint wavelet transformation to the output of FISTA
    F = representation_operator_trans(f_wav(1:N) + 1i*f_wav(N+1:end));
    F = reshape(F, [m,m]);
  
    %%%%% IMPORTANT__ HOW TO OCMPUTE PSNR WITH COMPLEX VALUE?? ¿?
    % Compute PSNR
    PSNR_list = [PSNR_list; abs(psnr(f, F))];

    %% VISUALIZE results
    fig = figure;
    fontsize = 16;

    imagesc(abs(F),[min(abs(F(:))), max(abs(F(:)))])
    t = strcat(strcat('Regularizer = ', num2str(regularization_parameter_lasso)), strcat(' PSNR = ', num2str(psnr_f)));
    title(t,'fontsize',fontsize,'interpreter','latex');
    file_name  = strcat('Images-33/', num2str(k));
    saveas(fig, file_name, 'epsc');
    
    % Set warm start
    x0 = f_wav; 
end

%% Visualize effect of regularizer parameter
fig = figure;

xlabel('Regularizer \lambda')
ylabel('PSNR (dB)')
semilogx( regs, PSNR_list,'LineWidth', 2)
legend('l1 regularization')

title('PSNR vs. Regularizer','fontsize',fontsize,'interpreter','latex');
file_name  = 'psnr-regularizer';
saveas(fig, file_name, 'epsc');
