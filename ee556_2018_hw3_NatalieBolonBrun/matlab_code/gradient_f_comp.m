function grad = gradient_f_comp(b, forward_operator, adjoint_operator, x, ind)
%Computes Gradient of f function
N_ind   = size(b,1)/2;
p       = size(x, 1)/2;

y_real  = b(1:N_ind);
y_comp  = b(N_ind+1:end);


f1      = y_real - real(forward_operator(x(1:p), ind)) + imag(forward_operator(x(p+1:end), ind));
f2      = y_comp - imag(forward_operator(x(1:p), ind)) - real(forward_operator(x(p+1:end), ind));

grad1      = [real(adjoint_operator(f1, ind)); imag(adjoint_operator(f1, ind))];
grad2      = [-imag(adjoint_operator(f2, ind)); real(adjoint_operator(f2, ind))];


grad    =  -grad1 -grad2;

end