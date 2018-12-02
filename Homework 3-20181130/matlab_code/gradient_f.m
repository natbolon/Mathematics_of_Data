function grad = gradient_f(y, forward_operator, adjoint_operator, x)
%Computes Gradient of f function
N_ind   = size(y,1)/2;
m       = size(x, 2);
y_real  = y(1:N_ind);
y_comp  = y(N_ind+1:end);

f1      = y_real - real(forward_operator(x(1:m,:))) + imag(forward_operator(x(m+1:end, :)));
f2      = y_comp - imag(forward_operator(x(1:m,:))) - real(forward_operator(x(m+1:end, :)));

A1      = [real(adjoint_operator(f1)); -imag(adjoint_operator(f1))];
A2      = [imag(adjoint_operator(f2)); real(adjoint_operator(f2))]; 


grad    = -A1 - A2;

end

