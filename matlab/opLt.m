function Z = opLt(x)
%% Ladj     opérateur adjoint du Laplacien
%     Z = Ladj(X)	calcule l'adjoint du Laplacien
%
%     Autrement dit, Ladj(X) calcule 
%     le produit matriciel :
%           L'*x 
%
%     Correspond à :
%     . l'adjoint de la dérivée seconde si X est un signal,
%     . l'adjoint du Laplacien V4 2D si X est une image non vectorisée.
%     
%
%     Ex 1 (2D):
%       im = im2double(imread('cameraman.tif'));
%       z  = Ladj(im);
% 
%       figure(1); clf;
%       subplot(121); imshow(im,[]);     title('image originale');
%       subplot(122); imshow(z,[]);      title('adjoint du Laplacien de l''image');
%%

dim = (min(size(x)) > 1) + 1;

switch dim
    case 1
        ker = [1 -2 1]';
        if isrow(x), ker = ker'; end
        Z = real(ifft(conj(psf2otf(ker,size(x))).*fft(x)));
        
    case 2
        ker = [0 1 0; 1 -4 1; 0 1 0]; % V4
        %ker = [1 1 1; 1 -8 1; 1 1 1]; % V8
        Z = real(ifft2(conj(psf2otf(ker,size(x))).*fft2(x)));
end




end