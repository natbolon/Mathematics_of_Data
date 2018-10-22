%*******************  EE556 - Mathematics of Data  ************************
% Function:  [x, info] = LSAGD (fx, gradf, parameter)
% Purpose:   Implementation of AGD with line search.     
% Parameter: parameter.x0         - Initial estimate.
%            parameter.maxit      - Maximum number of iterations.
%            parameter.Lips       - Lipschitz constant for gradient.
%			 parameter.strcnvx	- strong convexity parameter
%            fx                 - objective function
%            gradf              - gradient mapping of objective function
%*************************** LIONS@EPFL ***********************************
function [x, info] = LSAGD(fx, gradf, parameter)

    fprintf('%s\n', repmat('*', 1, 68));
    fprintf('Accelerated Gradient with line search\n')
    
    % Initialize x, y, t and L0.
    x = parameter.x0;
    y = parameter.x0;
    t = 0;
    L = parameter.Lips;

    % Main loop.
    for iter = 1:parameter.maxit
                           
        % Start the clock.
        tic;
        
        % Update the next iteration. (main algorithmic steps here!)
        % Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
        d = -gradf(y);
        L0 = 0.5*L;
        i = 0;
        
        while fx(y + (1/(L0*2^i))*d) > fx(y) - (1/(L0*2^(i+1)))*norm(d)^2
            i = i+1;
        end
        
        x_next = y + (1/(L0*2^i))*d;
        t_next = 0.5*(1+sqrt(1+2^(i+1)*t^2));
        y_next = x_next + (t/t_next)*(x_next-x);
        L = L0*2^i;
        
                      
        % Compute error and save data to be plotted later on.
        info.itertime(iter ,1)  = toc;
        info.fx(iter, 1)        = fx(x);
        
        % Print the information.
        if mod(iter, 5) == 0 || iter == 1
        fprintf('Iter = %4d, f(x) = %0.9f\n', ...
                iter, info.fx(iter, 1));
        end

        % Prepare the next iteration
        x     = x_next;
        t     = t_next;
        y     = y_next;

    end

    % Finalization.
    info.iter           = iter;

end
%**************************************************************************
% END OF THE IMPLEMENTATION.
%**************************************************************************
