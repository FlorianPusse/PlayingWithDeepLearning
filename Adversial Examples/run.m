function [ pred, x_grad, b1_grad,w1_grad,b2_grad,w2_grad,b3_grad,w3_grad] = run( x,t,b1,w1,b2,w2,b3,w3 )
    h0 = x;

    a1 = b1 + w1 * h0;
    h1 = sigmoid(a1);
    h1_d = h1.*(1-h1);

    a2 = b2 + w2 * h1;
    h2 = sigmoid(a2);
    h2_d = h2.*(1-h2);

    a3 = b3 + w3 * h2;
    h3 = sigmoid(a3);
    h3_d = h3.*(1-h3);

    % pred : 10 x 1
    pred = h3;

    g = pred - t;

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

