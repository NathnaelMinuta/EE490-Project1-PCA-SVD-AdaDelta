%% PCA / Singular Value Decomposition - Image Compression (SVD)
clc; close all;

% ===== Read image =====
I0 = imread("C:\Users\dn8659vx\Downloads\EE490\Project 1\dataset\For_Img_Compression.JPG");   % RGB uint8
R = I0(:,:,1); G = I0(:,:,2); B = I0(:,:,3);

% ===== Choose k values to explore =====
k_list = [10 20 40 60 80 120 160];   % explore these
k_best = 80;                          % your final chosen k (edit if needed)

% For PSNR computation
I0d = im2double(I0);
[m,n,~] = size(I0);

%% ===== Final result (Original vs Best k) =====
R1 = svd_compress_channel(R, k_best);
G1 = svd_compress_channel(G, k_best);
B1 = svd_compress_channel(B, k_best);
I_best = cat(3, R1, G1, B1);

figure;
subplot(1,2,1); imshow(I0);     title('Original');
subplot(1,2,2); imshow(I_best); title(sprintf('Compressed (k=%d)', k_best));

%% ===== Grid figure: all k in one screenshot + table =====
results = zeros(length(k_list), 3); % [k, PSNR_dB, CompressionRatio]

figure;
for i = 1:length(k_list)
    k = min(k_list(i), min(m,n));   % safety

    % Compress with this k
    Rk = svd_compress_channel(R, k);
    Gk = svd_compress_channel(G, k);
    Bk = svd_compress_channel(B, k);
    Ik = cat(3, Rk, Gk, Bk);

    % PSNR vs original
    Ikd = im2double(Ik);
    mse = mean((I0d(:) - Ikd(:)).^2);
    psnr_val = 10*log10(1/mse);

    % Theoretical compression ratio (raw storage)
    % Original: 3*m*n values
    % Compressed: 3*k*(m+n+1) values
    CR = (3*m*n) / (3*k*(m+n+1));

    results(i,:) = [k, psnr_val, CR];

    % Plot in grid (2 rows x 4 cols fits up to 8 k values)
    subplot(2,4,i);
    imshow(Ik);
    title(sprintf('k=%d\nPSNR=%.1f dB\nCR=%.1fx', k, psnr_val, CR));
end
sgtitle('SVD Compression Results for Different k');

% Print results table
T = array2table(results, 'VariableNames', {'k','PSNR_dB','CompressionRatio'});
disp(T);

%% ===== Helper Function =====
function C_out = svd_compress_channel(C_in, k)
    % Convert to double for SVD math
    C = double(C_in);

    % SVD decomposition
    [U,S,V] = svd(C, 'econ');

    % Ensure valid k
    k = min(k, min(size(S)));

    % Rank-k reconstruction (U_k * S_k * V_k^T)
    Ck = U(:,1:k) * S(1:k,1:k) * V(:,1:k)';

    % Clip to valid image range and return uint8
    Ck = min(max(Ck, 0), 255);
    C_out = uint8(Ck);
end
