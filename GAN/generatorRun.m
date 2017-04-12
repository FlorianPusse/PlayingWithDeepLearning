function [ pred, x_grad, b1_grad,w1_grad,b2_grad,w2_grad] = generatorRun( x,b1_G,w1_G,b2_G,w2_G,b1_D,w1_D,b2_D,w2_D,b3_D,w3_D,b4_D,w4_D,early)
    h0_G = x;
    
    a1_G = b1_G + w1_G * h0_G;
    h1_G = softplus(a1_G);
    h1_d_G = exp(a1_G) ./ (exp(a1_G) + 1);

    a2_G = b2_G + w2_G * h1_G;
    h2_G = a2_G;
    h2_d_G = ones(size(h2_G));

    pred_G = h2_G;
    
    [ pred, discriminator_grad ,~,~,~,~] = discriminatorRun(pred_G,b1_D,w1_D,b2_D,w2_D,b3_D,w3_D,b4_D,w4_D,early);

    g_G = discriminator_grad;
    
    g_G = g_G .* h2_d_G;
    b2_grad = g_G;
    w2_grad = g_G*h1_G';
    g_G = w2_G'*g_G;

    g_G = g_G .* h1_d_G;
    b1_grad = g_G;
    w1_grad = g_G*h0_G';
    x_grad = w1_G'*g_G;  
end