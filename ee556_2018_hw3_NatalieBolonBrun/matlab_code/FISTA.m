function [f_wav, theta_k] = FISTA(fx, gx, gradf, proxg, x0, Lips, maxit, tolx)
    
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
        x_new = proxg(y - lambda*gradf(y), gamma);
        
        % Check restart conditions
        if fx(x_new)+gx(x_new) - fx(x_k)- gx(x_k) > tolx
            
            theta_k     = 1;
            theta_old   = 1;
            y           = x_k;
            x_new       = proxg(y - lambda*gradf(y), gamma);
        end
        
        theta_old   = theta_k;
        theta_k     = (sqrt(theta_k^4 + 4*theta_k^2) - theta_k^2)/2;
        x_old       = x_k;
        x_k         = x_new;
       

    end
    
    f_wav = x_k;
end

