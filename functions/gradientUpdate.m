function [] = gradientUpdate( biases, weights, bias_gradients, weight_gradients, training_rate, doAscend )
    L = max(cell2mat(keys(weights)));
    
    for l = 1:L
        if (doAscend)
            biases(l) = biases(l) + training_rate*bias_gradients(l);
            weights(l) = weights(l) + training_rate*weight_gradients(l);
        else
            biases(l) = biases(l) - training_rate*bias_gradients(l);
            weights(l) = weights(l) - training_rate*weight_gradients(l);
        end
    end
end

