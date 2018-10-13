%*******************  EE556 - Mathematics of Data  ************************
% This function returns:
%  fx: objective function, i.e., fx(x) evaluate value of f at x
%  gradf: gradient mapping, i.e., gradf(x) evaluates gradient of f at x
%  gradfsto: stochastic gradient mapping, i.e., gradfsto(x,i) evaluates
%               gradient of f_i at x
%  hessf: Hessian mapping, i.e., hessf(x) evaluates hessian of f at x
%*************************** LIONS@EPFL ***********************************

function [ fx, gradf, gradfsto, hessfx ] = Oracles( b, A, sigma, h )

    function fx = fmodifiedhuber(A, b, x, h)
       
        n = size(A,1);
        fx = zeros(n,1);
        
        for i = 1:n
           
            yf = b(i) * A(i,:) * x;
            fx(i) = ( abs( 1 - yf ) <= h ) * (( 1 + h - yf )^2 / ( 4 * h )) + ( yf < 1 - h ) * ( 1 - yf );
            
        end
        
        fx = 0.5 * mean(fx);
        
    end


    function gradfx = gradfmodifiedhuber(A, b, x, h)
       
        n = size(A,1);
        gradfx = zeros(size(x));
        for i = 1:n
           
            yf = b(i) * A(i,:) * x;
            if abs( 1 - yf ) <= h
                gradfx = gradfx + ( ( yf - 1 - h ) / ( 2 * h ) ) * b(i) * A(i,:)';
            elseif yf < 1 - h
                gradfx = gradfx - b(i) * A(i,:)';
            end
            
        end

        gradfx = gradfx / (2 * n);
            
    end

    
    function hessfx = hessfmodifiedhuber(A, b, x, h)
    
        [n, p] = size(A);
        hessfx = zeros(p, p);
        
        for i = 1:n
           
            if abs( 1 - b(i) * A(i,:) * x ) <= h
                hessfx = hessfx + A(i,:)' * A(i,:);
            end
            
        end
        
        hessfx = hessfx / ( 4 * n * h );
    
    end

    
    function stogradfx = stogradfmodifiedhuber(A, b, x, h, i)
    
        yf = b(i) * A(i,:) * x;
        
        
        if abs( 1 - yf ) <= h
            stogradfx = ( ( yf - 1 - h ) / ( 2 * h ) ) * b(i) * A(i,:)';
        elseif yf < 1 - h
            stogradfx = -1 * b(i) * A(i,:)';
        else
            stogradfx = zeros(size(x));
        end
        
    end


    [n, p] = size(A);
    
    fx      = @(x)(0.5*sigma*norm(x)^2 + fmodifiedhuber(A, b, x, h));
    gradf  = @(x)(sigma*x + gradfmodifiedhuber(A, b, x, h));
    gradfsto  = @(x, i)(sigma*x + stogradfmodifiedhuber(A, b, x, h, i));
    hessfx  = @(x)(sigma*eye(p) + hessfmodifiedhuber(A, b, x, h));

end

%**************************************************************************
% END OF THE IMPLEMENTATION.
%**************************************************************************
