%*******************  EE556 - Mathematics of Data  ************************
% Function:  [x, info] = SVR(fx, gradf, gradfsto, parameter)       
% Purpose:   Implementation of the stochastic gradient descent algorithm with variance reduction.     
% Parameter: parameter.x0         - Initial estimate.
%            parameter.maxit      - Maximum number of iterations.
%            parameter.L          - Maximum of Lipschitz constant for gradients.
%            parameter.strcnvx    - Strong convexity parameter of f(x).
%            parameter.no0functions - number of functions
%            fx                 - objective function
%            gradfsto           - stochastic gradient mapping of objective function
%            gradf              - gradient mapping of objective function
%*************************** LIONS@EPFL ***********************************
function [x, info] = SVR(fx, gradf, gradfsto, parameter)
 
    fprintf('%s\n', repmat('*', 1, 68));
    fprintf('Stochastic Gradient Descent with variance reduction\n')
        
        
    % Initialize.
    x = parameter.x0;
    n = parameter.no0functions;
    gamma = 0.01/parameter.Lips;
    q = round(1000*parameter.Lips,0);
    
    % Main loop.
    for iter    = 1:parameter.maxit
            
        % Start timer
        tic;
       
        % Update the next iteration. (main algorithmic steps here!)
        % Use the notation xb_next for x_{k+1}, and xb for x_{k}, and similar for other variables.
        
        v = gradf(x);
        size_x = size(x);
        x_bar_l = zeros(size_x(1),q);
        x_bar_l(:,1) = x;
        for l = 1:q
            i = randi(n);
            v = gradfsto(x_bar_l(:,l),i) - gradfsto(x,i) + v;
            x_bar_l(:,l) = x_bar_l(:,l) - gamma*v;
        end
        
        x_next = (1/q)*(x_bar_l*ones(q,1));
        
        % Compute error and save data to be plotted later on.
        info.itertime(iter ,1)      = toc;
        info.fx(iter, 1)            = fx(x);
        
         % Print the information.
        if mod(iter, 5) == 0 || iter == 1
        fprintf('Iter = %4d,  f(x) = %0.9f\n', ...
                iter,  info.fx(iter, 1));
        end
        
        % Prepare the next iteration
        x = x_next;
 
    end
 
    % Finalization.
    info.iter           = iter;
    
end
%**************************************************************************
% END OF THE IMPLEMENTATION.
%**************************************************************************?
