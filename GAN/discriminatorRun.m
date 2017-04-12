function [ pred, x_grad, b1_grad,w1_grad,b2_grad,w2_grad,b3_grad,w3_grad,b4_grad,w4_grad] = discriminatorRun( x,b1,w1,b2,w2,b3,w3,b4,w4,real)
    h0 = x;
    
    a1 = b1 + w1 * h0;
    h1 = tanh(a1);
    h1_d = 1-(h1.^2);

    a2 = b2 + w2 * h1;
    h2 = tanh(a2);
    h2_d = 1-(h2.^2);
    
    a3 = b3 + w3 * h2;
    h3 = tanh(a3);
    h3_d = 1-(h3.^2);
    
    a4 = b4 + w4 * h3;
    h4 = sigmoid(a4);
    h4_d = h4 .* (1-h4);

    pred = h4;

    if(real)
        if(pred == 0)
           pred = 0.000000000001; 
        end
        g = 1 ./ pred;
    else
        if(pred == 1)
           pred = 0.999999999999; 
        end
        g = 1 ./ (pred-1);
    end
    
    g = g .* h4_d;
    b4_grad = g;
    w4_grad = g*h3';
    g = w4'*g;

    g = g .* h3_d;
    b3_grad = g;
    w3_grad = g*h2';
    g = w3'*g;
    
    g = g .* h2_d;
    b2_grad = g;
    w2_grad = g*h1';
    g = w2'*g;

    g = g .* h1_d;
    b1_grad = g;
    w1_grad = g*h0';
    x_grad = w1'*g;
end

