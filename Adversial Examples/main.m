addpath('../data/mnist');
addpath('../functions');

X = loadMNISTImages('train-images.idx3-ubyte')';
Y = loadMNISTLabels('train-labels.idx1-ubyte');
Xtest = loadMNISTImages('t10k-images.idx3-ubyte')';
Ytest = loadMNISTLabels('t10k-labels.idx1-ubyte');
Y = Y + 1;
Ytest = Ytest + 1;

% epsilon as defined in the paper
epsilon = 0.15;

% number of epochs
epochs = 10;

% number of samples
n = size(X,1);

% number of test samples
nTest = size(Xtest,1);

% dimensions
d = size(X,2);

% step size
alpha = 0.2;

batchsize = 100;
batches = n/batchsize;

[ biases, weights ] = createNetwork( [784,256,50,10], 0.1 );

for epoch = 1 : epochs
    for b = 1 : batches        
        interval = (b-1)*batchsize + 1:b*batchsize;
        X_tmp = X(interval,:);
        Y_tmp = Y(interval,:);
        
        [ ~, ~, bias_gradients, weight_gradients] = run( X_tmp, biases, weights,@sigmoid_activation,@sigmoid_activation,@(x) x-encodeY(Y_tmp), false);
        gradientUpdate( biases, weights, bias_gradients, weight_gradients, alpha, false )
    end
    
    if epoch + 1 < epochs
        pred = run( Xtest, biases, weights,@sigmoid_activation,@sigmoid_activation,NaN, true);
    else
        [ pred, x_grad, ~, ~] = run( Xtest, biases, weights,@sigmoid_activation,@sigmoid_activation,@(x) x-encodeY(Ytest), false);
    end
    [~, pred] = max(pred);
    correct = sum(pred' == Ytest);
    
    display(strcat('Done epoch: ',num2str(epoch), '. Error rate: ',num2str(1 - correct/nTest)));
end;

% store which samples are classified correctly
correctlyClassified = (pred' == Ytest);

% get sign of gradient
gradient_sign = sign(x_grad');

% calculate pertubated test samples
modifiedX = Xtest + gradient_sign*epsilon;
modifiedX(modifiedX < 0) = 0;
modifiedX(modifiedX > 1) = 1;

% predict labels of modified samples
 modified_pred = run( modifiedX, biases, weights,@sigmoid_activation,@sigmoid_activation, NaN, true);
[~, modified_pred] = max(modified_pred);

fuckedUp = (modified_pred' ~= Ytest) & correctlyClassified;
display(strcat('Error rate with adversial examples: ',num2str(1 - sum(modified_pred' == Ytest)/nTest)));

% get fucked up examples
allIndices = find(fuckedUp == 1);
nrExamples = length(allIndices);
chosenIndices = randi(nrExamples,20);

figure;
for i = 1:3:30
    subplot(10,3,i);
    imshow(reshape(Xtest(chosenIndices(i),:),[28,28]));
    subplot(10,3,i+1);
    imshow(reshape(gradient_sign(chosenIndices(i),:),[28,28]));
    subplot(10,3,i+2);
    imshow(reshape(modifiedX(chosenIndices(i),:),[28,28]));
end