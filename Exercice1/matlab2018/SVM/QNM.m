%*******************  EE556 - Mathematics of Data  ************************
% Function:  [x, info] = QNM (fx, gradf, hessf, parameter)
% Purpose:   Implementation of the quasi-Newton method with BFGS update.     
% Parameter: parameter.x0         - Initial estimate.
%            parameter.maxit      - Maximum number of iterations.
%            parameter.Lips       - Lipschitz constant for gradient.
%			 parameter.strcnvx	- strong convexity parameter
%            fx                 - objective function
%            gradf              - gradient mapping of objective function
%*************************** LIONS@EPFL ***********************************
function [x, info] = QNM(fx, gradf, parameter)
    
    fprintf('%s\n', repmat('*', 1, 68));
    fprintf('Quasi Newton Method\n')

    % Initialize x, B0, alpha, grad (and any other)
	alpha     = 10;
    kappa     = .1;
    x         = parameter.x0;
    Bk        = eye(numel(x));
    grad      = gradf(x);
    fval      = fx(x);
	    
    % Main loop.
    for iter = 1:parameter.maxit
        
        % Start the clock.
        tic;
        
        % Update the next iteration. (main algorithmic steps here!)
        % Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
        
         %%%% YOUR CODES HERE
        
        % Compute error and save data to be plotted later on.
        info.itertime(iter ,1)  = toc;
        info.fx(iter, 1)        = fx(x);
                
        % Print the information.
        fprintf('Iter = %4d, f(x) = %0.9f\n', ...
                iter, info.fx(iter, 1));
                
        % Prepare the next iteration
        x     = x_next;

    end

    % Finalization.
    info.iter           = iter;
    
end
%**************************************************************************
% END OF THE IMPLEMENTATION.
%**************************************************************************
