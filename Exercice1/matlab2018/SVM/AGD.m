%*******************  EE556 - Mathematics of Data  ************************
% Function:  [x, info] = AGD (fx, gradf, parameter)
% Purpose:   Implementation of the accelerated gradient descent algorithm.     
% Parameter: parameter.x0         - Initial estimate.
%            parameter.maxit      - Maximum number of iterations.
%            parameter.Lips       - Lipschitz constant for gradient.
%			 parameter.strcnvx	- strong convexity parameter
%            fx                 - objective function
%            gradf              - gradient mapping of objective function
%*************************** LIONS@EPFL ***********************************
function [x, info] = AGD(fx, gradf, parameter)
    
    fprintf('%s\n', repmat('*', 1, 68));
    fprintf('Accelerated Gradient\n')
    
    % Initialize x, y and t.
    
     x = parameter.x0;
     t = 0;
     y = parameter.x0;
     alpha = 1/parameter.Lips;
     

    % Main loop.
    for iter = 1:parameter.maxit
        
        % Start the clock.
        tic;        
        
        % Update the next iteration. (main algorithmic steps here!)
        % Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
        x_next = y - alpha*gradf(y);
        t_next = 0.5*(1 + sqrt(1 + 4*(t^2)));
        y_next = x_next + (t-1)*(x_next-x)/(t_next);
        
        % Compute error and save data to be plotted later on.
        info.itertime(iter ,1)  = toc;
        info.fx(iter, 1)        = fx(x);
                
        % Print the information.
        if mod(iter, 5) == 0 || iter == 1
        fprintf('Iter = %4d, f(x) = %0.9f\n', ...
                iter, info.fx(iter, 1));
        end
        
        % Prepare next iteration
        x           = x_next;
        t           = t_next;
        y           = y_next;
        
    end

    % Finalization.
    info.iter           = iter;
    
end
%**************************************************************************
% END OF THE IMPLEMENTATION.
%**************************************************************************
