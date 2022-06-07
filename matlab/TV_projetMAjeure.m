clear variables
close all
clc

%% param
IMAGE = rgb2gray(im2double(imread("color3.jpg")));
[h,w] = size(IMAGE);
z = IMAGE(floor(h/2),:);
z = z(:);
lambda = 10;
gamma = 0.05;

figure(1)
imshow(IMAGE,[]);
figure(2)
plot(z);

%% choix de l'opérateur de régularisation
gamma_choix = 'laplacien';
switch gamma_choix
    case 'id'
        G =@(x) x;
        Gt =@(x) x;
    case 'gradient'
        G = @(x) opD(x);
        Gt =@(x) opDt(x);
    case 'laplacien'
        G = @(x) opL(x);
        Gt =@(x) opLt(x);
end

%% calcul de la fonction à minimiser
f =@(x) norm(x(:),2);
g =@(x) lambda*norm(x(:),1);


%% calcul du prox_gamma_f(x)
%prox_g =@(x)(x>lambda*gamma).*(x-lambda*gamma) + (x<-lambda*gamma).*(x+lambda*gamma);

prox_tau_g =@(x,tau) (x>tau).*(x-tau) + (x<-tau).*(x+tau);
gradF =@(u) -G(-Gt(u)+z);

%% Méthode de descente
flag = true;
epsilon = 1e-4;
iteration = 1;
uk = G(z);
ENERGIE = [];
while iteration < 60000
    % Iteartion proximale
    uk1 = uk - gamma*gradF(uk) - gamma*prox_tau_g( uk/gamma - gradF(uk) ,lambda/gamma);   
    uk = uk1;
    iteration = iteration + 1;

end

uhat = uk;
xhat = -Gt(uhat) + z;
%%
figure(2);hold on;
plot(xhat,'r')

figure(3)
hold on
plot(diff(xhat))
plot(diff(diff(xhat)))
legend("diff", "diffdiff")
