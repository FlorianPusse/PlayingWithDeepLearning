function [g, grad] = softplus_activation(z)
  g = log(exp(z) + 1);
  grad = exp(g) ./ (exp(g) + 1);
end