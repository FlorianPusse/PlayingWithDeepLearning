function [ x_gradient ] = J_grad_discriminator( pred, discriminator_biases, discriminator_weights )
    [~, x_gradient, ~, ~] = run(pred, discriminator_biases, discriminator_weights,@tanh_activation,@sigmoid_activation,@(x) 1./(x-1), false);
end

