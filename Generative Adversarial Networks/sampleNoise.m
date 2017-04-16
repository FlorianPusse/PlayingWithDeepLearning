function [ z ] = sampleNoise(n)
    z = linspace(-8,8,n)' + rand([n,1]) * 0.01;
end

