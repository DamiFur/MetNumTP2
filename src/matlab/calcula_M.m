function [Train, T, m, n, mu, Mu, X, M, aval] = calcula_M
% Carga la matriz de train
tic
Train = dlmread('../../data/train.csv', ',', 1, 0);
toc
tic
dlmwrite('train', Train, 'delimiter', ' ');
toc


% Se queda solo con las imagenes
tic
T = Train(:, 2:size(Train,2));
toc

n = size(T, 1);
m = size(T, 2);

% Calcula la media
tic
mu = mean(T);
toc
% Calcula una matriz de [mu, mu, .. , mu]
tic
Mu = repmat(mu, size(Train,1), 1);
toc

% Genera X
tic
X = T - Mu;
toc
tic
dlmwrite('x', X, 'delimiter', ' ');
dlmwrite('xt', X', 'delimiter', ' ');
toc

% Genera M
tic
M = (1/(n -1)) * (X' * X);
toc

% escribe M
tic
dlmwrite('xtx', M, 'delimiter', ' ' );
toc

% calcula los autovalores
tic
aval = eig(M)
toc

