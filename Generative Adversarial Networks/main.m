addpath('../functions');

mean = 4;
dev = 0.5;

% stepsize used for pretraining
pretrain_alpha = 0.10;

% number of epochs to do pretraining
pretraining_epochs = 1000;

% stepsize
alpha = 0.03;

% number of dimensions for the noise
noise_dim = 1;

% number of dimensions for the input of the discriminator
input_dim = 1;

% number of nodes per hidden layer in discriminator
hidden_size_D = 8;

% number of nodes per hidden layer in generator
hidden_size_G = 4;

% number of epochs
epochs = 1500;

% number of predictions to do after each epoch
predictionSize = 1000;

% standart deviation used for normal initalization of weights
init_dev = 0.1;

% batchsize
batchsize = 100;

% number of times the descriminator is updated for 
% one update of the generator
k = 3;

[ discriminator_biases, discriminator_weights ] = createNetwork( [input_dim,hidden_size_D,hidden_size_D,hidden_size_D,1], 0.1 );
[ generator_biases, generator_weights ] = createNetwork( [noise_dim,hidden_size_G,input_dim], 0.1 );

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
    
    [ ~, ~, bias_gradients, weight_gradients ] = run( randValues, discriminator_biases, discriminator_weights,@tanh_activation,@sigmoid_activation,@(x) x-probabilities', false);
    gradientUpdate( discriminator_biases, discriminator_weights, bias_gradients, weight_gradients, pretrain_alpha, false );
end

for epoch = 1:epochs
    for k_it = 1:k        
        % sample from data distribution
        realData = sort(normrnd(mean,dev,[batchsize,1]));
        [ ~, ~, bias_gradients, weight_gradients ] = run( realData, discriminator_biases, discriminator_weights,@tanh_activation,@sigmoid_activation,@(x) 1./x, false);
        
        % sample from noise distribution
        noise_D = sampleNoise(batchsize);
        
        % apply generator to noise
        generatedData = run( noise_D, generator_biases, generator_weights,@softplus_activation,@identity_activation, NaN, true);
        generatedData = sort(generatedData)';
        [ ~, ~, bias_gradients2, weight_gradients2 ] = run( generatedData, discriminator_biases, discriminator_weights,@tanh_activation,@sigmoid_activation,@(x) 1./(x-1), false);     
        
        gradientUpdate( discriminator_biases, discriminator_weights, bias_gradients, weight_gradients, alpha, true );
        gradientUpdate( discriminator_biases, discriminator_weights, bias_gradients2, weight_gradients2, alpha, true );
    end;
    
    % sample from noise to update generator
    noise_G = sampleNoise(batchsize);
    % apply generator to noise  
    [ ~, ~, bias_gradients, weight_gradients ] = run( noise_G, generator_biases, generator_weights,@softplus_activation,@identity_activation,@(pred) J_grad_discriminator( pred',discriminator_biases, discriminator_weights), false);
    gradientUpdate( generator_biases, generator_weights, bias_gradients, weight_gradients, alpha, false );
    
    % sample from noise distribution
    noise = sampleNoise(predictionSize);
    predictions(:,epoch) = run( noise, generator_biases, generator_weights,@softplus_activation,@identity_activation, NaN, true);
    
    % sample for discriminator
    indices = -8:0.01:8;
    disc(:,epoch) = run(indices',discriminator_biases, discriminator_weights,@tanh_activation,@sigmoid_activation,NaN, true);
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