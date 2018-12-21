function [x_k, error_vals, F_vals] = FISTA_non_monotonicity(fx, gx, gradf, proxg, x0, Lips, maxit, F_star)
    
    format long
    % Initialize parameters
    theta_k     = 1;
    theta_old   = 1;
    
    x_old       = x0; 
    x_k         = x0; 
    
    lambda      = 1/Lips;   % Step size
    gamma       = 1/Lips;

    error_vals  = [];
    F           = fx(x0) + gx(x0);
    F_vals      = [F];
    
    % Iterate
    for k=1:maxit
        if rem(k,50) == 0
            disp(strcat('Iteration: ', num2str(k)))
        end
        
        y     = x_k + theta_k*((1/theta_old) -1)*(x_k-x_old);
        x_new = proxg(y - lambda*gradf(y), gamma);
        
        theta_old   = theta_k;
        theta_k     = (sqrt(theta_k^4 + 4*theta_k^2) - theta_k^2)/2;
        
        % Check restart conditions
        if fx(x_new) + gx(x_new )  > F 
            theta_k     = 1;
            theta_old   = 1;
            x_new       = proxg(x_k - lambda*gradf(x_k), gamma);
        end
        
        x_old       = x_k;
        x_k         = x_new;
        
        % Store values
        F           = fx(x_k) + gx(x_k);
        F_vals      = [F_vals; F];
        error       = ( F - F_star)/F_star;
        error_vals  = [error_vals, error]; 
        
        % Stop iterating
        %if abs(F-F_vals(k))/(F_vals(k)) < 1e-15
        %    disp('Reached tolerance')
        %    break
        %end
        if error < 1e-15
            'Reached Tolerance'
            break
        end
        
        

    end
   
end
