function [ biases, weights ] = createNetwork( layer_sizes, deviation )
    weights = containers.Map('KeyType','uint32','ValueType','any');
    biases = containers.Map('KeyType','uint32','ValueType','any');
    for i = 2:length(layer_sizes)
        lastSize = layer_sizes(i-1);
        size = layer_sizes(i);
        weights(i-1) = normrnd(0,deviation,[size,lastSize]);
        biases(i-1) = zeros(size,1);
    end
end

