function [ x_gradient ] = J_grad_discriminator( pred, discriminator_biases, discriminator_weights, mode )
    N = max(cell2mat(keys(discriminator_biases)));
    predictions = zeros(N,length(pred));
    gradients = zeros(N,length(pred));
    for i = 1:N;
        [predictions(i,:), gradients(i,:), ~, ~] = run(pred, discriminator_biases(i), discriminator_weights(i),@tanh_activation,@sigmoid_activation,@(x) 1./(x-1), false);
    end

    if(strcmp(mode,'max'))
        [~, indices] = max(predictions);
        idx = sub2ind(size(gradients), indices, 1:size(gradients, 2));
        x_gradient = gradients(idx);
    else
        % use simple mean of discriminators
        x_gradient = sum(gradients);
        x_gradient = x_gradient ./ double(N);
    end
end

