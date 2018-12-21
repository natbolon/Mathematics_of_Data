function [f_wav, theta_k] = FISTA_comparison(fx, gx, gradf, proxg, x0, Lips, maxit, tolx, b ,ind)
    p = size(ind);

    % Initialize parameters
    theta_k     = 1;
    theta_old   = 1;
    
    x_old       = x0; 
    x_k         = x0; 
   

    lambda      = 1/Lips;   % Step size
    %gamma       = lambda;   % Parameter for proximal operator
    gamma = lambda;
    
    % Iterate
    for k=1:maxit
       
        y     = x_k + theta_k*((1/theta_old) -1)*(x_k-x_old);
        x_new = proxg(y - lambda*gradf(y, b, ind), gamma);
        theta_old   = theta_k;
        theta_k     = (sqrt(theta_k^4 + 4*theta_k^2) - theta_k^2)/2;
        
        % Check restart conditions
        %disp(strcat('F_new = ', num2str(fx(x_new, b, ind, p)+gx(x_new))))
        %disp(strcat('f-next ', num2str(fx(x_new, b, ind, p))))
        %disp(strcat('g-next ', num2str( gx(x_new))))
        if fx(x_new, b, ind, p)+gx(x_new) > fx(x_k, b, ind, p)+ gx(x_k)  
            disp('restart');
            theta_k     = 1;
            theta_old   = 1;
            y           = x_k;
            x_new       = proxg(y - lambda*gradf(y, b, ind), gamma);
            
        end
        
        x_old       = x_k;
        x_k         = x_new;
       

    end
    
    f_wav = x_k;
end

