addpath('../functions');

mean = 4;
dev = 0.5;

% stepsize used for pretraining
pretrain_alpha = 0.10;

% number of epochs to do pretraining
pretraining_epochs = 1000;

% stepsize
initial_learning_rate = 0.03;
decay_rate = 0.95;
decay_steps = 100;

% number of dimensions for the noise
noise_dim = 1;

% number of dimensions for the input of the discriminator
input_dim = 1;

% number of nodes per hidden layer in discriminator
hidden_size_D = 8;

% number of nodes per hidden layer in generator
hidden_size_G = 4;

% number of epochs
epochs = 5000;

% number of predictions to do after each epoch
predictionSize = 1000;

% standart deviation used for normal initalization of weights
init_dev = 0.1;

% batchsize
batchsize = 100;

% number of times the descriminator is updated for 
% one update of the generator
k = 3;

N = 3;

discriminator_biases = containers.Map('KeyType','uint32','ValueType','any');
discriminator_weights = containers.Map('KeyType','uint32','ValueType','any');

 for i = 1:N
     [ discriminator_biases(i), discriminator_weights(i) ] = createNetwork( [input_dim,hidden_size_D,hidden_size_D,hidden_size_D,1], 0.3 );
end

[ generator_biases, generator_weights ] = createNetwork( [noise_dim,hidden_size_G,input_dim], 0.1 );

% plot the original distribution
figure;
histfit(normrnd(mean,dev,1000,1));

% only used for storing values to display later on
predictions = zeros(predictionSize,epochs);
disc = cell(1,N);
for i = 1:N
    disc{i} = zeros(length(-8:0.01:8),epochs);
end

for epoch = 1:epochs
    alpha = initial_learning_rate * decay_rate ^ (epoch/decay_steps);
    
    for k_it = 1:k        
        % sample from data distribution
        realData = sort(normrnd(mean,dev,[batchsize,1]));
        
        % sample from noise distribution
        noise_D = sampleNoise(batchsize);
        
        % apply generator to noise
        generatedData = run( noise_D, generator_biases, generator_weights,@softplus_activation,@identity_activation, NaN, true);
        generatedData = sort(generatedData)';
        
        for i = 1 : N
            [ ~, ~, bias_gradients, weight_gradients ] = run( realData, discriminator_biases(i), discriminator_weights(i),@tanh_activation,@sigmoid_activation,@(x) 1./x, false);
            [ ~, ~, bias_gradients2, weight_gradients2 ] = run( generatedData, discriminator_biases(i), discriminator_weights(i),@tanh_activation,@sigmoid_activation,@(x) 1./(x-1), false);     

            gradientUpdate( discriminator_biases(i), discriminator_weights(i), bias_gradients, weight_gradients, alpha, true );
            gradientUpdate( discriminator_biases(i), discriminator_weights(i), bias_gradients2, weight_gradients2, alpha, true );
        end
    end
        
    % sample from noise to update generator
    noise_G = sampleNoise(batchsize);
    % apply generator to noise
    if epoch < 500
       mode = 'mean';
    else
       mode = 'max';
    end
    
    [ ~, ~, bias_gradients, weight_gradients ] = run( noise_G, generator_biases, generator_weights,@softplus_activation,@identity_activation,@(pred) J_grad_discriminator( pred',discriminator_biases, discriminator_weights,mode), false);
    gradientUpdate( generator_biases, generator_weights, bias_gradients, weight_gradients, alpha, false );
    
    % sample from noise distribution
    noise = sampleNoise(predictionSize);
    predictions(:,epoch) = run( noise, generator_biases, generator_weights,@softplus_activation,@identity_activation, NaN, true);
    
    % sample for discriminator
    indices = -8:0.01:8;
    for i = 1:N
        disc{i}(:,epoch) = run(indices',discriminator_biases(i), discriminator_weights(i),@tanh_activation,@sigmoid_activation,NaN, true);
    end
end

p = 0;
if true
    figure;
    for e = 1:epochs
        subplot(1,2,1); 
        histogram(predictions(:,e));
        xlim([-1 8])
        title(num2str(e));
        if (p ~= 0)
            cla(p);
        end
        p = subplot(1,2,2);
        hold on;
        for i = 1:N
            plot(-8:0.01:8,disc{i}(:,e));
        end
        hold off;
        ylim([0 1])
        drawnow;
    end
end
histogram(predictions(:,epochs));