function [ output, grad ] = tanh_activation( x )
 output = tanh(x);
 grad = 1-(output.^2);
end

