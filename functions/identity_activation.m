function [ g, grad ] = identity_activation( x )
 g = x;
 grad = ones(size(g));
end

