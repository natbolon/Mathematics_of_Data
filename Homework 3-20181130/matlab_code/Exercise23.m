%% Problem size - image side
addpath('export_fig');
addpath('utilities/');
addpath('IMAGES/')

%% Load Image and transform
I           = imread('CS-image.jpg'); 
I           = rgb2gray(I);
I           = double(I);
I           = imresize(I, [256 256]);
y           = I + 15*randn(size(I)); % Generate noisy image

m_l1           = 256; 
N           = m_l1^2;

%% Wavelet operators
% Define the function handles that compute
% the products by W (DWT) and W' (inverse DWT)

wav         = daubcqf(8);
level       = log2(m_l1); % Maximum level

% Adjoint wavelet transform - From wavelet coefficients to image
WT          = @(x) midwt(x,wav,level); 
% Wavelet transform -  From image to wavelet coefficient
W           = @(x) mdwt(x,wav,level); 

% Define PSNR 
psnr_f      = @(I, I_trans) 20*log10(max(max(I))/sqrt(norm(I - I_trans, 'fro')));  


%% Lasso & TV approximations
prox_tv_maxiters = 100;
prox_tv_tol      = 1e-5;

psnr_l1          = [];
psnr_tv          = [];

regs             = linspace(1,20,50);

for i=1:length(regs)
    
    % Select regularizer value
    reg         = regs(i);
    
    % Compute l1 approximation
    alpha       = proxL1norm(W(y),reg);
    I_2         = WT(alpha);

    % Compute TV approximation 
    I_prox      = TV_prox(y, 'lambda', reg, 'maxiter', prox_tv_maxiters, 'tol', prox_tv_tol, 'verbose', 0);
    
    
    psnr_y      = psnr_f(I,y);
    
    psnr_l1     = [psnr_l1, psnr_f(I,I_2)];
    psnr_tv     = [psnr_tv, psnr_f(I, I_prox)];

end

%% Visualize effect of regularizer parameter
figure
hold on
xlabel('Regularizer \lambda')
ylabel('PSNR')

semilogx(regs, psnr_l1, 'LineWidth', 2)
semilogx(regs, psnr_tv, 'LineWidth', 2)
legend('l1 measure', 'tv measure')
hold off


%% Visualize Best Result
disp('Mx for l1')
[m_l1,i_l1] = max(psnr_l1)
reg_l1      = regs(i_l1)
alpha       = proxL1norm(W(y),reg_l1);
I_2         = WT(alpha);

disp('Mx for TV')
[m_tv, i_tv] = max(psnr_tv)
reg_tv       = regs(i_tv)
I_prox       = TV_prox(I, 'lambda', reg_tv, 'maxiter', prox_tv_maxiters, 'tol', prox_tv_tol, 'verbose', 0);



f = figure,
fontsize = 16;
% Original Image
%subplot(141),
imagesc(I,[min(I(:)), max(I(:))]), axis image off, colormap gray
title('Original','fontsize',fontsize,'interpreter','latex');
saveas(f, 'Images-best/original-l1.eps')

% Noisy Image
f = figure
% subplot(142)
imagesc(y,[min(y(:)), max(y(:))]), axis image off, colormap gray
title(strcat('NOISY - PSNR =',num2str(psnr_y)),'fontsize',fontsize,'interpreter','latex');
saveas(f, 'Images-best/noisy.eps')
% L1 image
f = figure
%subplot(143)
imagesc(I_2,[min(I_2(:)), max(I_2(:))]), axis image off, colormap gray
title(strcat('L1 - PSNR =',num2str(m_l1)),'fontsize',fontsize,'interpreter','latex');
saveas(f, 'Images-best/l1.eps')
% TV image
f = figure
%subplot(144)
imagesc(I_prox,[min(I_prox(:)), max(I_prox(:))]), axis image off, colormap gray
title(strcat('TV - PSNR =',num2str(m_tv)),'fontsize',fontsize,'interpreter','latex');
saveas(f, 'Images-best/tv.eps')


%% Effect very small and large regularizers l1

f = figure;
fontsize = 13;
% Original Image
imagesc(I,[min(I(:)), max(I(:))]), axis image off, colormap gray
%title('Original','fontsize',fontsize,'interpreter','latex');
saveas(f, 'original-l1.eps')

% Noisy
f = figure;
fontsize = 13;
imagesc(y,[min(y(:)), max(y(:))]), axis image off, colormap gray
%title('Noisy ','fontsize',fontsize,'interpreter','latex');
saveas(f, 'noisy-l1.eps')
% Best
f = figure;
fontsize = 13;
reg = 9.5306;
% Compute L1 approximation
alpha       = proxL1norm(W(y),reg);
I_2         = WT(alpha);
imagesc(I_2,[min(I_2(:)), max(I_2(:))]), axis image off, colormap gray
%title('Best result L1 ','fontsize',fontsize,'interpreter','latex');
saveas(f, 'best-l1.eps')

regs = [1e-3, 1e-2, 1e2, 1e3, 1e4];

for i=1:length(regs)
    f = figure;
    fontsize = 13;
    reg         = regs(i);
    
    % Compute L1 approximation
    alpha       = proxL1norm(W(y),reg);
    I_2         = WT(alpha);
    imagesc(I_2,[min(I_2(:)), max(I_2(:))]), axis image off, colormap gray
    %title(strcat('Regularizer = ', num2str(reg)),'fontsize',fontsize,'interpreter','latex');
    saveas(f, strcat(num2str(reg), 'l1.eps'))
end

%% Effect very small and large regularizers TV


imagesc(I_prox,[min(I_prox(:)), max(I_prox(:))]), axis image off, colormap gray
%title('Best result TV ','fontsize',fontsize,'interpreter','latex');
saveas(f, 'best-tv.eps')


regs = [1e-3, 1e-2, 1e2, 1e3, 1e4];

for i=1:length(regs)
    f = figure;
    fontsize = 13;
    reg         = regs(i);
    
    % Compute TV approximation 
    I_prox      = TV_prox(y, 'lambda', reg, 'maxiter', prox_tv_maxiters, 'tol', prox_tv_tol, 'verbose', 0);

    
    imagesc(I_prox,[min(I_prox(:)), max(I_prox(:))]), axis image off, colormap gray
    %title(strcat('Regularizer = ', num2str(reg)),'fontsize',fontsize,'interpreter','latex');
    saveas(f, strcat(num2str(reg), 'tv.eps'))
end


    