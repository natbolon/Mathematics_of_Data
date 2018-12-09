function grad = gradient_f(b, forward_operator, adjoint_operator, x)
%Computes Gradient of f function
N_ind   = size(b,1)/2;
p       = size(x, 1)/2;
m       = sqrt(p);
y_real  = b(1:N_ind);
y_comp  = b(N_ind+1:end);


f1      = y_real - real(forward_operator(x(1:p))) + imag(forward_operator(x(p+1:end)));
f2      = y_comp - imag(forward_operator(x(1:p))) - real(forward_operator(x(p+1:end)));

grad1      = [real(adjoint_operator(f1)); imag(adjoint_operator(f1))];
grad2      = [-imag(adjoint_operator(f2)); real(adjoint_operator(f2)) ];


grad    =  -grad1 -grad2;

end

