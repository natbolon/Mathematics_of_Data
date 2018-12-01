function [f_wav, o2] = FISTA(fx, gx, gradf, proxg, x0, Lips, maxit, tolx)
    o2 = 0;
    x_old   = x0;
    x       = x0;
    t_old   = 1; 
    t       = 1;
    lambda   = 1/Lips; 
    gamma   = 1;
    %% CHECK IMPLEMENTATION PROXY! PROBLEM WITH MATCHING DIMENSIONS!! 
    
    
    for k=1:maxit
        t_next = (sqrt(t^4 + 4*t^2) - t^2)/2;
        y      = x + t*(t_old - 1)*(x - x_old);
        x_next = proxg(y - lambda*gradf(y), gamma);
        
        if (fx(x_next) + gx(x_next) - (fx(x) - gx(x))) > tolx
            t_old   = 1;
            t       = 1;
            y       = x;
            x_next  = proxg(y - lambda*gradf(y), gamma);
        end
        
        t_old  = t;
        t      = t_next;
        x_old  = x;
        x      = x_next;
        
    end
    f_wav = x;
end

