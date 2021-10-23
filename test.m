function main
    mx = 0;
    while (true)
        n = randi([5, 20]);
        %m = randi([5, 20]);
        m = n;

        Phi = normrnd(0, 1, [n, m]);
        [U, S, V] = svd(Phi);
        r = min(n, m);
        idx = randsample(r, randi(2 * floor(sqrt(r))));
        S(idx, idx) = 0;
        Phi = U * S * V';
        
        d = rand([m, 1]);
        d = d / sum(d);
        D = diag(d);

        d1 = rand([m, 1]);
        d1 = d1 / sum(d1);
        D1 = diag(d1);

        t = (Phi * D * Phi') * pinv(Phi * D1 * Phi');
        tt = Phi * D * inv(D1) * inv(Phi);
        s = svd(t);
        z = s(1) / max(d ./ d1);
        if (z > mx)
            norm(t - tt, 'fro')
            mx = z
        end
        continue
        
        %q = normrnd(0, 1, [m, 1]);
        q = rand([m, 1]);

        w = pinv(Phi * D * Phi') * Phi * D * q;
        
        P = Phi;
        r = rank(P);
        [U, S, V] = svd(P);
        sigma = S(r, r) * norm(D, 'fro');
        ratio = (w' * w) * sigma * sigma / (q' * D * q);
        if (ratio > mx)
            mx = ratio
        end
    end
    
function [Q, R] = mgson(X)
    % Modified Gram-Schmidt orthonormalization (numerical stable version of Gram-Schmidt algorithm) 
    % which produces the same result as [Q,R]=qr(X,0)
    % Written by Mo Chen (sth4nth@gmail.com).
    [d,n] = size(X);
    m = min(d,n);
    R = zeros(m,n);
    Q = zeros(d,m);
    for i = 1:m
        v = X(:,i);
        for j = 1:i-1
            R(j,i) = Q(:,j)'*v;
            v = v-R(j,i)*Q(:,j);
        end
        R(i,i) = norm(v);
        Q(:,i) = v/R(i,i);
    end
    R(:,m+1:n) = Q'*X(:,m+1:n);