%% PCA / Singular Value Decomposition - image denoising

clc; close all;

I0 = imread("C:\Users\dn8659vx\Downloads\EE490\Project 1\dataset\For_Img_Denosing.JPG");     % I0: 3D uint8 image
R = I0(:,:,1); G = I0(:,:,2); B = I0(:,:,3);    % extract R, G, B

% ===== Explore k (key parameter) =====
k_list = [10 20 40 60 80 120 160];   % try a range, then pick best
k = 60;                              % pick one for the final comparison

% ===== Denoise each channel using SVD rank-k approximation =====
R1 = svd_denoise_channel(R, k);
G1 = svd_denoise_channel(G, k);
B1 = svd_denoise_channel(B, k);

% convert new R1, G1, B1 back to RGB
I_rgb = cat(3, R1, G1, B1);

% ===== Compare original vs denoised =====
figure;
subplot(1,2,1); imshow(I0);    title('Original (Noisy)');
subplot(1,2,2); imshow(I_rgb); title(sprintf('Denoised (SVD rank-k), k=%d', k));

%% ===== OPTIONAL: Show k sweep (recommended for your report) =====
figure;
for i = 1:length(k_list)
    kk = k_list(i);
    Rk = svd_denoise_channel(R, kk);
    Gk = svd_denoise_channel(G, kk);
    Bk = svd_denoise_channel(B, kk);
    Ik = cat(3, Rk, Gk, Bk);

    subplot(2,4,i);
    imshow(Ik);
    title(['k = ', num2str(kk)]);
end
sgtitle('SVD Denoising Results for Different k');

%% ===== OPTIONAL: Plot singular values (helps justify k) =====
% Pick one channel (R) to show the singular value decay
Rdouble = double(R);
[~,S,~] = svd(Rdouble, 'econ');
s = diag(S);

figure;
plot(s, 'o-'); grid on;
xlabel('Index'); ylabel('Singular value');
title('Singular values of R channel (use elbow to justify k)');

%% ===== Helper function =====
function C_out = svd_denoise_channel(C_in, k)
    % Convert to double for SVD math
    C = double(C_in);

    % SVD (econ = faster)
    [U,S,V] = svd(C, 'econ');

    % Ensure k is valid
    k = min(k, min(size(S)));

    % Rank-k reconstruction (keep dominant components)
    Ck = U(:,1:k) * S(1:k,1:k) * V(:,1:k)';

    % Clip to valid range and convert back to uint8
    Ck = min(max(Ck, 0), 255);
    C_out = uint8(Ck);
end
