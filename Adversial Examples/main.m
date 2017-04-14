addpath('../data/mnist');
X = loadMNISTImages('train-images.idx3-ubyte')';
Y = loadMNISTLabels('train-labels.idx1-ubyte');
Xtest = loadMNISTImages('t10k-images.idx3-ubyte')';
Ytest = loadMNISTLabels('t10k-labels.idx1-ubyte');
Y = Y + 1;
Ytest = Ytest + 1;

% epsilon as defined in the paper
epsilon = 0.20;

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

% initalization of weights
tmp = 6/(sqrt(d+10));


% w1 : first layer weights: 256 x 784
w1 = 2*tmp*rand(256,784)-tmp;
% b1: first layer bias
b1 = zeros(256,1);
% w2 : second layer weights: 50 x 256
w2 = 2*tmp*rand(50,256)-tmp;
% b2: second layer bias
b2 = zeros(50,1);
% w2 : second layer weights: 10 x 50
w3 = 2*tmp*rand(10,50)-tmp;
% b2: second layer bias
b3 = zeros(10,1);

for epoch = 1 : epochs
    for b = 1 : batches
        acc_b1_grad = zeros(256,1);
        acc_w1_grad = zeros(256,784);
        acc_b2_grad = zeros(50,1);
        acc_w2_grad = zeros(50,256);
        acc_b3_grad = zeros(10,1);
        acc_w3_grad = zeros(10,50);
        
        interval = (b-1)*batchsize + 1:b*batchsize;
        X_tmp = X(interval,:);
        Y_tmp = Y(interval,:);
        
        [~, ~, b1_grad, w1_grad, b2_grad, w2_grad, b3_grad, w3_grad] = runVectorized(X_tmp,Y_tmp,b1,w1,b2,w2,b3,w3);.    
        
        b1 = b1 - alpha*b1_grad;
        w1 = w1 - alpha*w1_grad;
        b2 = b2 - alpha*b2_grad;
        w2 = w2 - alpha*w2_grad;
        b3 = b3 - alpha*b3_grad;
        w3 = w3 - alpha*w3_grad;
    end
    
    [pred, x_grad, ~,~,~,~,~,~] = runVectorized(Xtest,Ytest,b1,w1,b2,w2,b3,w3);
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
[modified_pred, ~, ~,~,~,~,~,~] = runVectorized(modifiedX,Ytest,b1,w1,b2,w2,b3,w3);
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