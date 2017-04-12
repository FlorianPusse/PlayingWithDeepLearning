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
        
        for i = 1:batchsize
            % x : input value: 784x1
            x = X((b-1)*batchsize + i,:)';
            
            % encode answer
            t = zeros(10,1);
            t(Y((b-1)*batchsize + i,1)) = 1;
            
            [pred, x_grad, b1_grad, w1_grad, b2_grad, w2_grad, b3_grad, w3_grad] = run(x,t,b1,w1,b2,w2,b3,w3);
            acc_b1_grad = acc_b1_grad + b1_grad;
            acc_w1_grad = acc_w1_grad + w1_grad;
            acc_b2_grad = acc_b2_grad + b2_grad;
            acc_w2_grad = acc_w2_grad + w2_grad;
            acc_b3_grad = acc_b3_grad + b3_grad;
            acc_w3_grad = acc_w3_grad + w3_grad;
        end;
        acc_b1_grad = acc_b1_grad ./ batchsize;
        acc_w1_grad = acc_w1_grad ./ batchsize;
        acc_b2_grad = acc_b2_grad ./ batchsize;
        acc_w2_grad = acc_w2_grad ./ batchsize;
        acc_b3_grad = acc_b3_grad ./ batchsize;
        acc_w3_grad = acc_w3_grad ./ batchsize;
        
        b1 = b1 - alpha*acc_b1_grad;
        w1 = w1 - alpha*acc_w1_grad;
        b2 = b2 - alpha*acc_b2_grad;
        w2 = w2 - alpha*acc_w2_grad;
        b3 = b3 - alpha*acc_b3_grad;
        w3 = w3 - alpha*acc_w3_grad;
    end
end;

% calculate error on test set
correct = 0;
for i = 1:nTest
    x = Xtest(i,:)';
    t = zeros(10,1);
    t(Ytest(i,1)) = 1;
    
    [pred, x_grad, ~,~,~,~,~,~] = run(x,t,b1,w1,b2,w2,b3,w3);
    [~, pred] = max(pred);
    correctLabel = Ytest(i,1);
    if(pred == Ytest(i,1))
        correct = correct + 1;
    end
end;
display(strcat('Done epoch: ',num2str(epoch), '. Error rate: ',num2str(1 - correct/nTest)));

correctlyClassified = zeros(nTest,1);
fuckedUp = zeros(nTest,1);

modifiedX = Xtest;
noise = zeros(size(Xtest));
correct = 0;
for i = 1:nTest
    x = Xtest(i,:)';
    t = zeros(10,1);
    t(Ytest(i,1)) = 1;
    
    [pred, x_grad, ~,~,~,~,~,~] = run(x,t,b1,w1,b2,w2,b3,w3);
    [~, pred] = max(pred);
    correctLabel = Ytest(i,1);
    if(pred == Ytest(i,1))
        correct = correct + 1;
        correctlyClassified(i) = 1;
    end
    modifiedX(i,:) = modifiedX(i,:) + sign(x_grad')*epsilon;
    noise(i,:) = sign(x_grad');
end;
modifiedX(modifiedX < 0) = 0;
modifiedX(modifiedX > 1) = 1;

correct = 0;
for i = 1:nTest
    
    x = modifiedX(i,:)';
    t = zeros(10,1);
    t(Ytest(i,1)) = 1;
    
    [pred, x_grad, ~,~,~,~,~,~] = run(x,t,b1,w1,b2,w2,b3,w3);
    [~, pred] = max(pred);
    correctLabel = Ytest(i,1);
    if(pred == Ytest(i,1))
        correct = correct + 1;
    end
    if(pred ~= Ytest(i,1) && correctlyClassified(i) == 1)
        fuckedUp(i) = 1;
    end
end;
display(strcat('Error rate with adversial examples: ',num2str(1 - correct/nTest)));

% get fucked up examples
allIndices = find(fuckedUp == 1);
nrExamples = length(allIndices);
chosenIndices = randi(nrExamples,20);

figure;
for i = 1:3:30
    subplot(10,3,i);
    imshow(reshape(Xtest(chosenIndices(i),:),[28,28]));
    subplot(10,3,i+1);
    imshow(reshape(noise(chosenIndices(i),:),[28,28]));
    subplot(10,3,i+2);
    imshow(reshape(modifiedX(chosenIndices(i),:),[28,28]));
end