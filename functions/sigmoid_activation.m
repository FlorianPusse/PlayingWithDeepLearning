function [g, grad] = sigmoid_activation(z)
  g = 1.0 ./ ( 1.0 + exp(-z));
  g(g == 0) = 10^(-20);
  g(g == 1) = 1 - 10^(-20);
  grad = g .* (1-g);
end