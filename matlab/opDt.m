function Z = opDt(x)
%% Dadj     opérateur adjoint du gradient par différences finies à droite
%     Z = Dadj(X)	calcule l'adjoint du gradient par différences finies
%     à droite
%
%     Autrement dit, Dadj(X) calcule le produit matriciel :   
%           D'*x 
%
%     Compatible :
%     . 1D si X est un signal,
%     . 2D si X est une image non vectorisée.
%
%
%     Ex 1 (2D):
%       im = im2double(imread('cameraman.tif'));
%       im_grad = D(im);
%       z = Dadj(im_grad);
%
%       figure(1); clf;
%       subplot(121); imshow(im,[]);       title('image originale');
%       subplot(122); imshow(z,[]);        title('divergence du gradient de l''image');
%%

dim = (min(size(x)) > 1) + 1;

switch dim
    case 1
        Z = - [x(1); x(2:end-1) - x(1:end-2); -x(end-1)] ./ 2.;
        
    case 2
        x1 = x(:,:,1);
        x2 = x(:,:,2);
        Z  = - [x1(:,1), x1(:,2:end-1) - x1(:,1:end-2), -x1(:,end-1)] ./ 2. ...
             - [x2(1,:); x2(2:end-1,:) - x2(1:end-2,:); -x2(end-1,:)] ./ 2.;
end





end