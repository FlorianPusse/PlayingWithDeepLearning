function [ pred, x_grad, b1_grad,w1_grad,b2_grad,w2_grad,b3_grad,w3_grad] = runVectorized( X,Y,b1,w1,b2,w2,b3,w3 )
    n = size(X,1);
    X = X';
    Y = encodeY(Y);
    h0 = X;

    a1 = repmat(b1,1,n) + w1 * h0;
    h1 = sigmoid(a1);
    h1_d = h1.*(1-h1);

    a2 = repmat(b2,1,n) + w2 * h1;
    h2 = sigmoid(a2);
    h2_d = h2.*(1-h2);

    a3 = repmat(b3,1,n) + w3 * h2;
    h3 = sigmoid(a3);
    h3_d = h3.*(1-h3);

    % pred : 10 x 1
    pred = h3;
    g = pred - Y;
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


    b1_grad = mean(b1_grad,2);
    b2_grad = mean(b2_grad,2);
    b3_grad = mean(b3_grad,2);

    w1_grad = w1_grad ./ n;
    w2_grad = w2_grad ./ n;
    w3_grad = w3_grad ./ n;
end