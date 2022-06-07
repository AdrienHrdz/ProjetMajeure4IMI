function grad = opD(x)
%% D     opérateur de gradient par différences finies à droite
%     GRAD = D(X)	calcule le gradient par différences finies
%     à droite, 
%
%     Autrement dit, D(X) calcule le produit matriciel :   
%           D*x 
%
%     Compatible :
%     . 1D si X est un signal,
%     . 2D si X est une image non vectorisée.
%
%     Ex 1 (1D): 
%       t = [-10:.1:10]';
%       s = sin(t);
%       s_grad = D(s);
% 
%       figure(1); clf;
%       plot(t,s,'-',t,s_grad,'--');
%       legend('signal original','signal dérivé');
%
%
%     Ex 2 (2D):
%       im = im2double(imread('cameraman.tif'));
%       im_grad = D(im);
%
%       im_gradx = im_grad(1:size(im_grad,1)/2,:);
%       im_grady = im_grad(size(im_grad,1)/2+1:end,:);
%
%       figure(2); clf;
%       subplot(131); imshow(im,[]);       title('image originale');
%       subplot(132); imshow(im_gradx,[]); title('Gradient selon x de l''image');
%       subplot(133); imshow(im_grady,[]); title('Gradient selon y de l''image');
%%



dim = (min(size(x)) > 1) + 1;

switch dim
    case 1
        grad = [x(2:end) - x(1:end-1) ; 0] ./ 2.;
        
    case 2
        Dx_im = [x(:,2:end) - x(:,1:end-1) , zeros(size(x,1),1)] ./ 2.;
        Dy_im = [x(2:end,:) - x(1:end-1,:) ; zeros(1,size(x,2))] ./ 2.;
        
        grad = cat(3,Dx_im,Dy_im);
end




end