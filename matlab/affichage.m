clear variables
close all
clc

%% 
load("normBenin.mat")
NormCarreBenin = NormCarre;

load("normMalsain.mat")
NormCarreMalsain = NormCarre;

figure(1)
subplot(121)
imshow(NormCarreBenin)
title('Benin')
colorbar
caxis([0 4.5e-3])
subplot(122)
imshow(NormCarreMalsain)
title('Malsain')
colorbar
caxis([0 4.5e-3])

%%
coupeBenin = NormCarreBenin(75,:);
coupeMalsain = NormCarreMalsain(65,:);

figure(2)
hold on
plot(coupeBenin, 'LineWidth',1.5)
plot(coupeMalsain, 'LineWidth',1.5)
legend('benin', 'malsain')