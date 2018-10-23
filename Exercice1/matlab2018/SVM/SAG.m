%*******************  EE556 - Mathematics of Data  ************************
% Function:  [x, info] = SAG(fx, gradf, parameter)       
% Purpose:   Implementation of the stochastic averaging gradient algorithm.     
% Parameter: parameter.x0         - Initial estimate.
%            parameter.maxit      - Maximum number of iterations.
%            parameter.L          - Maximum of Lipschitz constant for gradients.
%            parameter.strcnvx    - Strong convexity parameter of f(x).
%            parameter.no0functions - number of functions
%            fx                 - objective function
%            gradfsto           - stochastic gradient mapping of objective function
%*************************** LIONS@EPFL ***********************************
function [x, info] = SAG(fx, gradfsto, parameter)

    fprintf('%s\n', repmat('*', 1, 68));
    fprintf('Stochastic Gradient Descent with averaging\n')
        
        
    % Initialize.
    x = parameter.x0;
    size_x = size(x);
    n = parameter.no0functions;
    v = zeros(size_x(1), n);
    alpha = 1/(16*parameter.Lips);
	    
    % Main loop.    
    for iter    = 1:parameter.maxit
            
        % Start timer
        tic;
       
        % Update the next iteration. (main algorithmic steps here!)
        % Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
        i = randi(n);
        v(:,i) = gradfsto(x,i);
        %v*ones(n,1) --> sum of v vectors
        x_next = x - alpha/n * (v*ones(n,1));
       
        % Compute error and save data to be plotted later on.
        info.itertime(iter ,1)      = toc;
        info.fx(iter, 1)            = fx(x);
        
         % Print the information.
        if mod(iter, 5) == 0 || iter == 1
        fprintf('Iter = %4d,  f(x) = %0.9f\n', ...
                iter,  info.fx(iter, 1));
        end
        
        % prepare next iteration
        x = x_next;

    end

    % Finalization.
    info.iter           = iter;
    
end
%**************************************************************************
% END OF THE IMPLEMENTATION.
%**************************************************************************
