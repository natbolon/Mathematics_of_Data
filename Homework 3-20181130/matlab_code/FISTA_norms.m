function [f_wav, F_vals, j] = FISTA_norms(f_x, g_x, gradf, proxg, x0, Lips, maxit, tolx, regularizer)
    
    % Initialize parameters
    theta_k     = 1;
    theta_old   = 1;
    
    x_old       = x0; 
    x_k         = x0; 
   
    F_vals      = [f_x(x0)+g_x(x0)];
    lambda      = 1/Lips;   % Step size
    %gamma = lambda*regularizer;
    gamma = regularizer;
    j = 0;
    
    % Iterate
    for k=1:maxit
        if rem(k,50) == 0
            disp(strcat('Iteration: ', num2str(k)))
        end
        
        y     = x_k + theta_k*((1/theta_old) -1)*(x_k-x_old);
        x_new = proxg(y - lambda*gradf(y), gamma);
     
        
%         disp(strcat('F_new = ', num2str(f_x(x_new)+g_x(x_new))))
%         disp(strcat('f-next ', num2str(f_x(x_new))))
%         disp(strcat('g-next ', num2str( g_x(x_new))))
       
        % Check restart conditions
        if f_x(x_new)+g_x(x_new) > (f_x(x_k)+ g_x(x_k))
            j = j +1
            theta_k     = 1;
            theta_old   = 1;
            y           = x_k;
            x_new       = proxg(y - lambda*gradf(y), gamma);
        end
        
        theta_old   = theta_k;
        theta_k     = (sqrt(theta_k^4 + 4*theta_k^2) - theta_k^2)/2;
        x_old       = x_k;
        x_k         = x_new;
        
        F_vals = [F_vals, f_x(x_k)+g_x(x_k)];
       
        if  norm(f_x(x_k)+g_x(x_k)) < tolx
            disp('Reached tolerance')
            break
        end

    end
    
    f_wav = x_k;
end

