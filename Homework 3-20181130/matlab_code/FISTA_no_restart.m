function [x_k, error_vals] = FISTA_no_restart(fx, gx, gradf, proxg, Lips, x0, maxit, F_star)
    
    % Parameter initialization
    x_k     = x0;
    y       = x0;
    t       = 1; 
    alpha   = 1/Lips;
    gamma   = 1/Lips;
    
    error_vals = [];
    
    for k=1:maxit
        if rem(k,50) == 0
            disp(strcat('Iteration: ', num2str(k)))
        end
        x_next = proxg(y - alpha*gradf(y), gamma);
        t_next = 0.5*(1 + sqrt(4*t^2 +1));
        y      = x_next + (t-1)/(t_next) * (x_next - x_k); 
        
        % Store values
        error  = abs(fx(x_next) + gx(x_next) - F_star)/F_star;
        error_vals = [error_vals, error]; 
        
        % Stop iterating
        if  error < 1e-15
            disp('Reached tolerance')
            break
        end
        
        x_k = x_next;
        t = t_next;
    end
    
end

