function lap = opL(x)
%% L     opérateur Laplacien
%     LAP = L(X)	calcule le Laplacien
%
%     Autrement dit, L(X) calcule le produit 
%     matriciel :
%           L*x 
%
%     Correspond à :
%     . la dérivée seconde si X est un signal,
%     . le Laplacien V4 2D si X est une image non vectorisée.
%
%
%     Ex 1 (1D):
%       t = [-10:.1:10]';
%       s = sin(t);
%       s_lap = L(s);
%
%       figure(1); clf;
%       plot(t,s,'-',t,s_lap,'--');
%       legend('signal original','dérivée seconde');
%
%
%     Ex 2 (2D):
%       im = im2double(imread('cameraman.tif'));
%       im_lap = L(im);
%
%       figure(2); clf;
%       subplot(121); imshow(im,[]);     title('image originale');
%       subplot(122); imshow(im_lap,[]); title('Laplacien de l''image');
%%

dim = (min(size(x)) > 1) + 1;


switch dim
    case 1
        ker = [1 -2 1]';
        if isrow(x), ker = ker'; end
        lap = real(ifft(psf2otf(ker,size(x)).*fft(x)));
        
    case 2
        ker = [0 1 0; 1 -4 1; 0 1 0]; % V4
        %ker = [1 1 1; 1 -8 1; 1 1 1]; % V8
        lap = real(ifft2(psf2otf(ker,size(x)).*fft2(x)));
end


end