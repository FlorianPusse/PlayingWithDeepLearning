function [ Y_encoded ] = encodeY( Y )
    n = size(Y,1);
    Y_encoded = zeros(10,n);
    for i = 1:n
        Y_encoded(Y(i),i) = 1;
    end
end