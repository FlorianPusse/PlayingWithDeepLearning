function [ pred, x_gradient, bias_gradients, weight_gradients ] = run( X, biases, weights, hidden_activation, output_activation, J_grad, forwardOnly )
    n = size(X,1);
    X = X';
    
    a = containers.Map('KeyType','uint32','ValueType','any');
    
    h = containers.Map('KeyType','uint32','ValueType','any');
    h(0) = X;

    h_d = containers.Map('KeyType','uint32','ValueType','any');
    
    bias_gradients = containers.Map('KeyType','uint32','ValueType','any');
    weight_gradients = containers.Map('KeyType','uint32','ValueType','any');
    
    % get number of layers
    L = max(cell2mat(keys(weights)));
    
    for l = 1:L-1
        a(l) = repmat(biases(l),1,n) + weights(l) * h(l-1);
        [h(l), h_d(l)] = hidden_activation(a(l));
    end
    
    a(L) = repmat(biases(L),1,n) + weights(L) * h(L-1);
    [h(L), h_d(L)] = output_activation(a(L));
    
    pred = h(L);
    
    if(forwardOnly)
        return;
    end
    
    g = J_grad(pred);
    
    for l = L:-1:1
        g = g .* h_d(l);
        bias_gradients(l) = mean(g,2);
        weight_gradients(l) = (g*h(l-1)')./ n;
        g = weights(l)'*g;
    end
    
    x_gradient = g;
end

