function [ pred_G ] = generatorForward( X,b1,w1,b2,w2)
    n = size(X,1);
    X = X';
    
    h0 = X;
    a1 = repmat(b1,1,n) + w1 * h0;
    h1 = softplus(a1);

    a2 = repmat(b2,1,n) + w2 * h1;
    h2 = a2;

    pred_G = h2;
end