function [ pred, x_grad, b1_grad,w1_grad,b2_grad,w2_grad] = generatorRun( X,b1,w1,b2,w2,b1_D,w1_D,b2_D,w2_D,b3_D,w3_D,b4_D,w4_D,early)
    n = size(X,1);
    X = X';

    h0 = X;
    
    a1 = repmat(b1,1,n) + w1 * h0;
    h1 = softplus(a1);
    h1_d = exp(a1) ./ (exp(a1) + 1);

    a2 = repmat(b2,1,n) + w2 * h1;
    h2 = a2;
    h2_d = ones(size(h2));

    pred = h2;
    
    [ ~, discriminator_grad ,~,~,~,~] = discriminatorRun(pred',b1_D,w1_D,b2_D,w2_D,b3_D,w3_D,b4_D,w4_D,early);

    g = discriminator_grad;
    
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

    w1_grad = w1_grad ./ n;
    w2_grad = w2_grad ./ n;
end