%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Exercise 4.1         Reconstruction of missing parts of image with FISTA %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 %% Add paths
 clear all
 addpath('utilities/');
 addpath('IMAGES/')

 %% Load Image and transform
 m       = 1024;

 I       = imread('Randa.jpg'); 
 I       = rgb2gray(I);
 I       = double(I);
 I       = imresize(I, [m, m]);

 N       = m^2;
 rate    = 0.4;

 ind     = randperm(N);
 p       = round(rate*N);
 ind     = reshape(sort(ind(1:p)), [p,1]);

 mask    = zeros([N,1]);
 mask(ind) = 1;
 mask    = reshape(mask, [m,m]);



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

 maxit       = 5000;
 maxit_tv    = 30;
 tolx        = 1e-15;
 regs1       = logspace(-5,5,15);
 regstv      = logspace(-5,5,15);

 Lips        = 1;

 PSNR_1      = [];
 PSNR_TV     = [];


 for k=1:length(regs1)
     reg_lasso = regs1(k);
     reg_tv    = regstv(k);
     %% Define Operators Norm 1
     fx_1          = @(x) 0.5*norm(b - S(WT(x)),'fro')^2;

     gradf_1       = @(x) -W(b - S(WT(x)));

     gx_1          = @(x) reg_lasso*norm(reshape(x, [N,1]),1);

     proxg_1       = @(x, reg) proxL1norm(x, reg);


     %% Define Operators Norm TV
     prox_tv_maxiters = 100
     prox_tv_tol      = 1e-5

     fx_tv          = @(x) 0.5*norm(b - S(x),'fro')^2;

     gradf_tv       = @(x) -(b - S(x));

     gx_tv          = @(x) reg_tv * TV_norm(x, 'iso');

     proxg_tv       = @(x, reg) TV_prox(x, 'lambda', reg, 'maxiter', prox_tv_maxiters, 'tol', prox_tv_tol, 'verbose', 0);



     %% 
     x0        = zeros(m);


     % Execute FISTA for Norm1
     time_norm1    = tic;
     [f_norm1, F_vals, ~]  = FISTA_norms(fx_1, gx_1, gradf_1, proxg_1, x0, Lips, maxit, tolx, reg_lasso);
     time_norm1    = toc(time_norm1)


     % Execute FISTA for Norm TV
     time_normtv    = tic;
     [f_norm_tv, ~] = FISTA_norms(fx_tv, gx_tv, gradf_tv, proxg_tv, x0, Lips, maxit_tv, tolx, reg_tv);
     time_normtv    = toc(time_normtv)

     f_norm1_t        = WT(f_norm1);

     % Reshape_images
     f_norm1_t        = reshape(f_norm1_t, [m,m]);
     f_norm_tv      = reshape(f_norm_tv, [m,m]);

     % Compute PSNR
     PSNR_1    = [PSNR_1, psnr(I,f_norm1_t)];
     PSNR_TV   = [PSNR_TV, psnr(I, f_norm_tv)];


 fig = figure;
 fontsize = 16;
 imagesc(f_norm1_t), colormap gray
 t = strcat('Norm 1 regularization', strcat('PNSR = ', num2str(PSNR_1(k))));
 title(t,'fontsize',fontsize,'interpreter','latex');

 fig = figure;
 fontsize = 16;
 imagesc(f_norm_tv), colormap gray
 t = strcat('TV Norm regularization', strcat('PNSR = ', num2str(PSNR_TV(k))));
 title(t,'fontsize',fontsize,'interpreter','latex');


 end

 %% Visualize effect of regularizer parameter
 fig = figure;
 hold on
 xlabel('Regularizer \lambda')
 ylabel('PSNR')
 semilogx(regs1, PSNR_1, 'LineWidth', 2)
 semilogx(regstv, PSNR_TV, 'LineWidth', 2)
 legend('l1 measure')
