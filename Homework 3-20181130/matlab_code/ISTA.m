function [x, error_vals] = ISTA(fx, gx, gradf, proxg, x0, Lips, maxit, F_star)
    
    alpha = 1/Lips; 
    gamma = 1/Lips;
    x     = x0; 
    error_vals = [];
    
    for k=1:maxit
        if rem(k,50) == 0
            disp(strcat('Iteration: ', num2str(k)))
        end
        
        x = proxg(x - alpha*gradf(x), gamma);
        
        % Store values
        error = abs(fx(x) + gx(x) - F_star)/F_star;
        error_vals = [error_vals, error]; 
        % Stop iteration
        if  error < 1e-15
            disp('Reached tolerance')
            break
        end
    end

end

