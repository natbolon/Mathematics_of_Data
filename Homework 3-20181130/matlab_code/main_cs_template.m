%% Problem size - image side
addpath('export_fig');
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
measurement_backward            = @(x) reshape(fft2adj_without_fftshift(x,ind,m,m),[N,1]);  % F^T P_Omega ^T -> Adjoint Operator.

%% REVISAR - COMO TENER A_REAL Y A_IMAG SI TENEMOS UN OPERADOR Y NO UNA MATRIX ??¿?¿
% Define the overall operator
forward_operator                = @(x) measurement_forward(representation_operator_trans(x)); % A
adjoint_operator                = @(x) representation_operator(measurement_backward(x)); % A^T 

%%
Lips    = 1;

% Generate measurements on complex domain and transfer to real domain
y       = [real(measurement_forward(f)), imag(measurement_forward(f))];

% Optimization parameters
maxit = 100;
tolx  = 1e-5;

%% WAVELET RECOVERY
% regularization parameter
regularization_parameter_lasso  = 0.0005;
% Initial point
x0      = f;

fx      = @(x) 0.5*(norm(y(:,1) -real(forward_operator(real(x))) + imag(forward_operator(imag(x))),2)^2 ...
        + norm(y(:,2) -imag(forward_operator(real(x))) - real(forward_operator(imag(x))),2)^2);
gx      = @(x) regularization_parameter_lasso*norm(x,1);
proxg   = @(x, gamma) proxL1norm_complex(x, regularization_parameter_lasso*gamma);

gradf = @(x) fx(x);


time_wav    = tic;
[f_wav, ~]  = FISTA(fx, gx, gradf, proxg, x0, Lips, maxit, tolx);
time_wav    = toc(time_wav)

psnr_wav    = @(I, I_trans) 20*log10(max(max(I))/sqrt(norm(I - I_trans, 'fro')));
pnsr        = psnr_wav(f, f_wav);



%% VISUALIZE results
