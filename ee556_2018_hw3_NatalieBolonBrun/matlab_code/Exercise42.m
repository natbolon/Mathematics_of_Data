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
 ST          = @(x) mask'.*x;
 


%% Define Parameters 

b = S(I);

maxit       = 1000; 
tolx        = 1e-15;
reg_lasso   = 10;
Lips        = 1;



%% Define Operators Norm 1
fx          = @(x) 0.5*norm(b - S(WT(x)),'fro')^2;

gradf       = @(x) -W((b - S(WT(x))));

gx          = @(x) reg_lasso*norm(reshape(x, [N,1]),1);

proxg       = @(x, reg) proxL1norm(x, reg);

 F_star = fx(W(I)) + gx(W(I)); %Ground truth

 %F_star = 5.096030358114446e+07; %Optimal Value with FISTA
% non-monotonicity



%% Define Starting point
x0        = rand(m);



%% Execute ISTA 
disp('Executing ISTA')
time    = tic;
[f_ISTA, error_ISTA, F_vals]             = ISTA(fx, gx, gradf, proxg, x0, Lips, maxit, F_star);
disp(strcat('Time ISTA: ', num2str(toc(time))))

% Display Image
% fig = figure;
% fontsize = 16;
% imagesc(WT(f_ISTA)), colormap gray,  axis image off;
% t = 'ISTA';
% title(t,'fontsize',fontsize,'interpreter','latex');
% saveas(fig, strcat('Images-43/', t), 'eps');

%%  Execute FISTA without restart
disp('Executing FISTA without restart')
time    = tic;
[f_FISTA_nr, error_FISTA_nr]     = FISTA_no_restart(fx, gx, gradf, proxg, Lips, x0, maxit, F_star);
disp(strcat('Time: ', num2str(toc(time))))

% Display Image
% fig = figure;
% fontsize = 16;
% imagesc(WT(f_FISTA_nr)), colormap gray,  axis image off;
% t = 'FISTA without restart';
% title(t,'fontsize',fontsize,'interpreter','latex');
% saveas(fig, strcat('Images-43/', t), 'eps');

%% Execute FISTA fixed iteration restart
disp('Executing FISTA with restart every 20 iterations')
it_restart = 20;
time    = tic;
[f_FISTA_iter_20, error_FISTA_iter_20, F_20] = FISTA_iter_restart(fx, gx, gradf, proxg, x0, Lips, maxit, it_restart, F_star);
disp(strcat('Time: ', num2str(toc(time))))

% Display Image
% fig = figure;
% fontsize = 16;
% imagesc(WT(f_FISTA_iter_20)), colormap gray,  axis image off;
% t = 'FISTA with restart after 20 iterations';
% title(t,'fontsize',fontsize,'interpreter','latex');
% saveas(fig, strcat('Images-43/', t), 'eps');


disp('Executing FISTA with restart every 50 iterations')
it_restart = 50;
time    = tic;
[f_FISTA_iter_50, error_FISTA_iter_50, F_50] = FISTA_iter_restart(fx, gx, gradf, proxg, x0, Lips, maxit, it_restart, F_star);
disp(strcat('Time: ', num2str(toc(time))))

% Display Image
% fig = figure;
% fontsize = 16;
% imagesc(WT(f_FISTA_iter_50)), colormap gray,  axis image off;
% t = 'FISTA with restart after 50 iterations';
% title(t,'fontsize',fontsize,'interpreter','latex');
% saveas(fig, strcat('Images-43/', t), 'eps');

disp('Executing FISTA with restart every 100 iterations')
it_restart = 100;
time    = tic;
[f_FISTA_iter_100, error_FISTA_iter_100, F_100] = FISTA_iter_restart(fx, gx, gradf, proxg, x0, Lips, maxit, it_restart, F_star);
disp(strcat('Time: ', num2str(toc(time))))

% Display Image
% fig = figure;
% fontsize = 16;
% imagesc(WT(f_FISTA_iter_100)), colormap gray,  axis image off;
% t = 'FISTA with restart after 100 iterations';
% title(t,'fontsize',fontsize,'interpreter','latex');
% saveas(fig, strcat('Images-43/', t), 'eps');

disp('Executing FISTA with restart every 200 iterations')
it_restart = 200;
time    = tic;
[f_FISTA_iter_200, error_FISTA_iter_200, F_200] = FISTA_iter_restart(fx, gx, gradf, proxg, x0, Lips, maxit, it_restart, F_star);
disp(strcat('Time: ', num2str(toc(time))))

% Display Image
% fig = figure;
% fontsize = 16;
% imagesc(WT(f_FISTA_iter_200)), colormap gray,  axis image off;
% t = 'FISTA with restart after 200 iterations';
% title(t,'fontsize',fontsize,'interpreter','latex');
% saveas(fig, strcat('Images-43/', t), 'eps');

%% Execute FISTA with gradient scheme restart
disp('Executing FISTA with gradient scheme restart')
time    = tic;
[f_FISTA_grad, error_FISTA_grad, F_FISTA_grad] = FISTA_grad_restart(fx, gx, gradf, proxg, x0, Lips, maxit, F_star);
disp(strcat('Time: ', num2str(toc(time))))
fx(f_FISTA_grad) + gx(f_FISTA_grad);

% Display Image
% fig = figure;
% fontsize = 16;
% imagesc(WT(f_FISTA_grad)), colormap gray,  axis image off;
% t = 'FISTA with restart with gradient strategy';
% title(t,'fontsize',fontsize,'interpreter','latex');
% saveas(fig, strcat('Images-43/', t), 'eps');

% %% Execute FISTA with non-monotonicity scheme restart
% disp('Executing FISTA with non-monotonicity scheme restart ')
% time    = tic;
% [f_FISTA, error_FISTA, F_FISTA] = FISTA_non_monotonicity(fx, gx, gradf, proxg, x0, Lips, maxit, F_star);
% disp(strcat('Time: ', num2str(toc(time))))
% 
% figure
% plot(F_FISTA)
% fx(f_FISTA) + gx(f_FISTA)
% % % Visualize evolution of F
% % figure
% % plot(F_FISTA)
% % 
% % % Optimal value 
% % fx(f_FISTA) + gx(f_FISTA)
% % 
% % % Display Image
% % fig = figure;
% % fontsize = 16;
% % imagesc(WT(f_FISTA)), colormap gray,  axis image off;
% % t = 'FISTA with non monotonicity restart strategy';
% % title(t,'fontsize',fontsize,'interpreter','latex');
% % saveas(fig, strcat('Images-43/', t), 'eps')

 %% Visualize effect of regularizer parameter
fig = figure;
hold on
xlabel('Iterations')
ylabel('Error')
plot(error_ISTA, 'LineWidth',0.5)
plot(error_FISTA_nr, 'LineWidth', 2)
plot(error_FISTA_iter_20, 'o')
plot(error_FISTA_iter_50, '+')
plot(error_FISTA_iter_100, '*')
plot(error_FISTA_iter_200, '')
plot(error_FISTA_grad, '--', 'LineWidth',0.5)
% plot(error_FISTA)

legend('ISTA','FISTA no restart', 'FISTA 20 iter', 'FISTA 50 iter', 'FISTA 100 iter', 'FISTA 200 iter', 'FISTA gradient restart' )

