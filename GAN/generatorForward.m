function [ pred_G ] = generatorForward( x,b1,w1,b2,w2)
    h0 = x;
    a1 = b1 + w1 * h0;
    h1 = softplus(a1);

    a2 = b2 + w2 * h1;
    h2 = a2;

    pred_G = h2;
end