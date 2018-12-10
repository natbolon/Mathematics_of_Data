function [x_k, error_vals] = FISTA_iter_restart(fx, gx, gradf, proxg, x0, Lips, maxit, it_restart, F_star)
    
    % Initialize parameters
    theta_k     = 1;
    theta_old   = 1;
    
    x_old       = x0; 
    x_k         = x0; 
   

    lambda      = 1/Lips;   % Step size
    gamma       = 1/Lips;
    i           = 0; %Iteration counter for the restart
    
    error_vals = [];
    
    % Iterate
    for k=1:maxit
        if rem(k,50) == 0
            disp(strcat('Iteration: ', num2str(k)))
        end
        
        y     = x_k + theta_k*((1/theta_old) -1)*(x_k-x_old);
        x_new = proxg(y - lambda*gradf(y), gamma);
        
        % Check restart conditions
        if i == it_restart
            theta_k     = 1;
            theta_old   = 1;
            y           = x_k;
            x_new       = proxg(y - lambda*gradf(y), gamma);
            i = 0;
        end
        
        theta_old   = theta_k;
        theta_k     = (sqrt(theta_k^4 + 4*theta_k^2) - theta_k^2)/2;
        x_old       = x_k;
        x_k         = x_new;
        i = i+1;

        % Store values
        error = abs(fx(x_k) + gx(x_k) - F_star)/F_star;
        error_vals = [error_vals, error]; 
       
        % Stop iterating
        if  error < 1e-15
            disp('Reached tolerance')
            break
        end
       

    end

end
