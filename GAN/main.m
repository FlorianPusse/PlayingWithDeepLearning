mean = 4;
dev = 0.5;

% stepsize used for pretraining
pretrain_alpha = 0.10;

% number of epochs to do pretraining
pretraining_epochs = 1000;

% stepsize
alpha = 0.007;

% number of dimensions for the noise
noise_dim = 1;

% number of nodes per hidden layer in discriminator
hidden_size_D = 8;

% number of nodes per hidden layer in generator
hidden_size_G = 4;

% number of epochs
epochs = 5000;

% number of predictions to do after each epoch
predictionSize = 1000;

% standart deviation used for normal initalization of weights
init_dev = 0.2;

% batchsize
batchsize = 100;

% number of times the descriminator is updated for 
% one update of the generator
k = 3;

% w1 : first layer weights: hidden_size_D x noise_dim
w1_D = normrnd(0,init_dev,[hidden_size_D,1]);
% b1: first layer bias
b1_D = zeros(hidden_size_D,1);
% w2 : second layer weights: 1 x hidden_size_D
w2_D = normrnd(0,init_dev,[hidden_size_D,hidden_size_D]);
% b2: second layer bias
b2_D = zeros(hidden_size_D,1);
% w2 : second layer weights: 1 x hidden_size_D
w3_D = normrnd(0,init_dev,[hidden_size_D,hidden_size_D]);
% b2: second layer bias
b3_D = zeros(hidden_size_D,1);
% w3 : second layer weights: 1 x hidden_size_G
w4_D = normrnd(0,init_dev,[1,hidden_size_D]);
% b3: second layer bias
b4_D = zeros(1,1);

% w1 : first layer weights: hidden_size_G x input_dim
w1_G = normrnd(0,init_dev,[hidden_size_G,1]);
% b1: first layer bias
b1_G = zeros(hidden_size_G,1);
% w2 : second layer weights: output_dim x hidden_size_G
w2_G = normrnd(0,init_dev,[1,hidden_size_G]);
% b2: second layer bias
b2_G = zeros(1,1);

% plot the original distribution
figure;
histfit(normrnd(mean,dev,1000,1));

% only used for storing values to display later on
predictions = zeros(predictionSize,epochs);
disc = zeros(length(-8:0.01:8),epochs);

for epoch = 1:pretraining_epochs    
    % sample from range (-7,7) and check the probability of samples values
    randValues = sort(2*(mean+2)*rand([batchsize,1])-(mean+2));
    probabilities = normpdf(randValues,mean,dev);
    
    [ ~, ~, b1_D_grad,w1_D_grad,b2_D_grad,w2_D_grad,b3_D_grad,w3_D_grad,b4_D_grad,w4_D_grad] = pretrainRun( randValues ,b1_D,w1_D,b2_D,w2_D,b3_D,w3_D,b4_D,w4_D,probabilities);
    
    b1_D = b1_D - pretrain_alpha*b1_D_grad;
    w1_D = w1_D - pretrain_alpha*w1_D_grad;
    b2_D = b2_D - pretrain_alpha*b2_D_grad;
    w2_D = w2_D - pretrain_alpha*w2_D_grad;
    b3_D = b3_D - pretrain_alpha*b3_D_grad;
    w3_D = w3_D - pretrain_alpha*w3_D_grad;
    b4_D = b4_D - pretrain_alpha*b4_D_grad;
    w4_D = w4_D - pretrain_alpha*w4_D_grad;
end

for epoch = 1:epochs
    for k_it = 1:k        
        % sample from data distribution
        realData = sort(normrnd(mean,dev,[batchsize,1]));
        
        % sample from noise distribution
        noise_D = sampleNoise(batchsize);
        
        % apply generator to noise
        generatedData = generatorForward(noise_D,b1_G,w1_G,b2_G,w2_G);
        generatedData = sort(generatedData)';
        
        [ ~, ~, b1_D_grad,w1_D_grad,b2_D_grad,w2_D_grad,b3_D_grad,w3_D_grad,b4_D_grad,w4_D_grad] = discriminatorRun( realData ,b1_D,w1_D,b2_D,w2_D,b3_D,w3_D,b4_D,w4_D,true);
        [ ~, ~, b1_D2_grad,w1_D2_grad,b2_D2_grad,w2_D2_grad,b3_D2_grad,w3_D2_grad,b4_D2_grad,w4_D2_grad] = discriminatorRun( generatedData ,b1_D,w1_D,b2_D,w2_D,b3_D,w3_D,b4_D,w4_D,false);        
        
        b1_D = b1_D + alpha*(b1_D_grad + b1_D2_grad);
        w1_D = w1_D + alpha*(w1_D_grad + w1_D2_grad);
        b2_D = b2_D + alpha*(b2_D_grad + b2_D2_grad);
        w2_D = w2_D + alpha*(w2_D_grad + w2_D2_grad);
        b3_D = b3_D + alpha*(b3_D_grad + b3_D2_grad);
        w3_D = w3_D + alpha*(w3_D_grad + w3_D2_grad);
        b4_D = b4_D + alpha*(b4_D_grad + b4_D2_grad);
        w4_D = w4_D + alpha*(w4_D_grad + w4_D2_grad); 
    end;
    
    noise_G = sampleNoise(batchsize);
    % apply generator to noise
    generatedData = generatorForward(noise_G,b1_G,w1_G,b2_G,w2_G);
    generatedData = sort(generatedData);
    
    [ ~, ~, b1_G_grad,w1_G_grad,b2_G_grad,w2_G_grad] = generatorRun( noise_G,b1_G,w1_G,b2_G,w2_G,b1_D,w1_D,b2_D,w2_D,b3_D,w3_D,b4_D,w4_D,false);
    
    b1_G = b1_G - alpha*b1_G_grad;
    w1_G = w1_G - alpha*w1_G_grad;
    b2_G = b2_G - alpha*b2_G_grad;
    w2_G = w2_G - alpha*w2_G_grad;
    
    % sample from noise distribution
    noise = sampleNoise(predictionSize);
    [pred] = generatorForward(noise,b1_G,w1_G,b2_G,w2_G);
    predictions(:,epoch) = pred;

    % sample for discriminator
    indices = -8:0.01:8;
    [ disc(:,epoch), x_grad, b1_grad,w1_grad,b2_grad,w2_grad,b3_grad,w3_grad,b4_grad,w4_grad] = discriminatorRun(indices',b1_D,w1_D,b2_D,w2_D,b3_D,w3_D,b4_D,w4_D,true);
end

if true
    figure;
    for e = 1:epochs
        subplot(1,2,1); 
        histogram(predictions(:,e));
        xlim([-1 8])
        title(num2str(e));
        subplot(1,2,2); 
        plot(-8:0.01:8,disc(:,e));
        ylim([0 1])
        drawnow;
    end
end
histogram(predictions(:,epochs));