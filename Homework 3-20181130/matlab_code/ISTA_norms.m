function [x] = ISTA_norms(gradf, proxg, x0, Lips, maxit)
    
    alpha = 1/Lips; 
    x     = x0; 
    
    for i=1:maxit
        x = proxg(x + alpha*gradf(x), 1);
    end

end

