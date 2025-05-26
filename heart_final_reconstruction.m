% Load in saved points
data1 = load('points_frame1.mat'); % Load struct
data2 = load('points_frame2.mat');
data3 = load('points_frame3.mat');

% Extract the point data
points_frame1 = data1.points_frame1;
points_frame2 = data2.points_frame2;
points_frame3 = data3.points_frame3;

% Match manually selected points between frames
disp('Using manually picked points for matching.');

% Match points between frame 1 and frame 2
matched_points1_12 = points_frame1;
matched_points2_12 = points_frame2;

% Match points between frame 1 and frame 3
matched_points1_13 = points_frame1;
matched_points3_13 = points_frame3;

% Verify the sizes of the matched points
if size(matched_points1_12, 2) < 2 || size(matched_points2_12, 2) < 2 || ...
   size(matched_points1_13, 2) < 2 || size(matched_points3_13, 2) < 2
    error('Point arrays must have at least two columns (x, y coordinates).');
end

% Plot matches for frame1-frame2
figure;
imshowpair(frame1, frame2, 'montage');
hold on;

% Plot matched points on the left image
plot(matched_points1_12(:, 1), matched_points1_12(:, 2), 'ro');

% Plot matched points on the right image, shifted by the width of frame 1
plot(matched_points2_12(:, 1) + size(frame1, 2), matched_points2_12(:, 2), 'go');

% Draw lines between matched points
for i = 1:size(matched_points1_12, 1)
    x1 = matched_points1_12(i, 1);
    y1 = matched_points1_12(i, 2);
    x2 = matched_points2_12(i, 1) + size(frame1, 2); % Shift x by the width of frame 1
    y2 = matched_points2_12(i, 2);
    plot([x1 x2], [y1 y2], 'b-');
end
title('Matched Points Between Frame 1 and Frame 2');
hold off;

% Plot matches for frame1-frame3
figure;
imshowpair(frame1, frame3, 'montage');
hold on;

% Plot matched points on the left image
plot(matched_points1_13(:, 1), matched_points1_13(:, 2), 'ro');

% Plot matched points on the right image, shifted by the width of frame 1
plot(matched_points3_13(:, 1) + size(frame1, 2), matched_points3_13(:, 2), 'go');

% Draw lines between matched points
for i = 1:size(matched_points1_13, 1)
    x1 = matched_points1_13(i, 1);
    y1 = matched_points1_13(i, 2);
    x2 = matched_points3_13(i, 1) + size(frame1, 2); % Shift x by the width of frame 1
    y2 = matched_points3_13(i, 2);
    plot([x1 x2], [y1 y2], 'b-');
end
title('Matched Points Between Frame 1 and Frame 3');
hold off;

% Save matched points
save('matched_points1_12.mat', 'matched_points1_12');
save('matched_points2_12.mat', 'matched_points2_12');
save('matched_points1_13.mat', 'matched_points1_13');
save('matched_points3_13.mat', 'matched_points3_13');

disp('Manual matching completed and saved.');





% CREATE DISPARITY MAP

% Ensure matched points are loaded
load('matched_points1_12.mat'); % Matched points between frame 1 and frame 2
load('matched_points2_12.mat');
load('matched_points1_13.mat'); % Matched points between frame 1 and frame 3
load('matched_points3_13.mat');

% Image dimensions (for disparity map size)
[h, w, ~] = size(frame1);

% Initialize sparse disparity arrays
sparse_disparity_12 = nan(h, w); % Disparity map between frame1 and frame2
sparse_disparity_13 = nan(h, w); % Disparity map between frame1 and frame3

% Populate sparse disparity maps
for i = 1:size(matched_points1_12, 1)
    x1 = round(matched_points1_12(i, 1));
    y1 = round(matched_points1_12(i, 2));
    x2 = round(matched_points2_12(i, 1));
    disparity = x1 - x2;
    if x1 > 0 && y1 > 0 && x1 <= w && y1 <= h
        sparse_disparity_12(y1, x1) = disparity;
    end
end

for i = 1:size(matched_points1_13, 1)
    x1 = round(matched_points1_13(i, 1));
    y1 = round(matched_points1_13(i, 2));
    x3 = round(matched_points3_13(i, 1));
    disparity = x1 - x3;
    if x1 > 0 && y1 > 0 && x1 <= w && y1 <= h
        sparse_disparity_13(y1, x1) = disparity;
    end
end

% Extract non-NaN points from the sparse disparity map
[y_sparse_12, x_sparse_12] = find(~isnan(sparse_disparity_12)); % Non-NaN indices
values_sparse_12 = sparse_disparity_12(~isnan(sparse_disparity_12)); % Corresponding disparity values

[y_sparse_13, x_sparse_13] = find(~isnan(sparse_disparity_13)); % Non-NaN indices
values_sparse_13 = sparse_disparity_13(~isnan(sparse_disparity_13)); % Corresponding disparity values

% Generate grid for dense disparity map
[x_grid, y_grid] = meshgrid(1:w, 1:h);

% Interpolate using griddata
dense_disparity_12 = griddata(x_sparse_12, y_sparse_12, values_sparse_12, x_grid, y_grid, 'cubic');
dense_disparity_13 = griddata(x_sparse_13, y_sparse_13, values_sparse_13, x_grid, y_grid, 'cubic');

% Plot sparse disparity maps
figure;
subplot(1, 2, 1);
imagesc(sparse_disparity_12);
title('Sparse Disparity Map (Frame 1 - Frame 2)');
colorbar;
colormap jet;

subplot(1, 2, 2);
imagesc(sparse_disparity_13);
title('Sparse Disparity Map (Frame 1 - Frame 3)');
colorbar;
colormap jet;

% Plot dense disparity maps
figure;
subplot(1, 2, 1);
imagesc(dense_disparity_12);
title('Dense Disparity Map (Frame 1 - Frame 2)');
colorbar;
colormap jet;

subplot(1, 2, 2);
imagesc(dense_disparity_13);
title('Dense Disparity Map (Frame 1 - Frame 3)');
colorbar;
colormap jet;



% CAMERA CALIBRATION
%{
fprintf('Calibrating the camera using checkerboard images...\n');

% Checkerboard dimensions
checkerboardSize = [10,7]; % Number of internal corners
squareSize = 30; % Square size in millimeters

% List of checkerboard images
checkerboardImages = {'pattern.png', 'check2.jpg'};

% Generate world coordinates for checkerboard corners
worldPoints = generateCheckerboardPoints(checkerboardSize, squareSize);


% Detect checkerboard points
[imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(checkerboardImages);

% Iterate through used images and display detected points
figure;
for i = 1:length(imagesUsed)
    if imagesUsed(i)
        % Load and read the corresponding image
        I = imread(checkerboardImages{i});
        
        % Overlay detected checkerboard points
        I_with_points = insertMarker(I, imagePoints(:, :, i), 'circle', ...
            'MarkerColor', 'green', 'Size', 30);
        
        % Display the image with detected points
        subplot(2, 2, i);
        imshow(I_with_points);
        title(sprintf('Checkerboard Points in Image %d', i));
    else
        fprintf('Checkerboard not detected in image: %s\n', checkerboardImages{i});
    end
end




% Check that the rows of imagePoints match worldPoints
if size(imagePoints, 1) ~= size(worldPoints, 1)
    error(['Mismatch: imagePoints has ', num2str(size(imagePoints, 1)), ...
           ' rows, but worldPoints has ', num2str(size(worldPoints, 1)), ' rows.']);
end

% Calibrate the camera
imageSize = size(img); % Use size of the last processed image
if length(imageSize) == 3
    imageSize = imageSize(1:2); % Extract height and width for RGB images
end

[cameraParams, ~] = estimateCameraParameters(imagePoints, worldPoints, ...
    'ImageSize', imageSize);

% Extract the intrinsic matrix K
K = cameraParams.IntrinsicMatrix';
disp('Intrinsic Matrix (K):');
disp(K);

% Visualize one of the checkerboard detections
figure;
validImage = validImageIndices(1); % Use the first valid image for visualization
imshow(checkerboardImages{validImage});
hold on;
plot(imagePoints(:, 1, 1), imagePoints(:, 2, 1), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
title('Checkerboard Detection in First Valid Image');
hold off;
%}


% ASSUME INTRINSIC MATRIX (K)
fprintf('Assuming intrinsic matrix (K)...\n');
f = 1000; % Assumed focal length in pixels
cx = 430 / 2; % Center of the width (image width divided by 2)
cy = 530 / 2; % Center of the height (image height divided by 2)


K = [1000,    0,  215;  % Updated cx
        0, 1000,  265;  % Updated cy
        0,    0,    1];
disp('Updated Intrinsic Matrix (K):');
disp(K);



% ESTIMATE FUNDAMENTAL MATRIX

% Generate the A matrix for frame 1 and frame 2
fprintf('Generating the A matrix for frame 1 and frame 2...\n');

n = size(matched_points1_12, 1); % Number of correspondences (Nc)
A_12 = zeros(n, 9); % Initialize A matrix for frame 1 and frame 2

for i = 1:n
    % Extract points from frame 1 and frame 2
    x = matched_points1_12(i, 1);
    y = matched_points1_12(i, 2);
    x_prime = matched_points2_12(i, 1);
    y_prime = matched_points2_12(i, 2);

    % Populate the A matrix
    A_12(i, :) = [x_prime * x, x_prime * y, x_prime, ...
                  y_prime * x, y_prime * y, y_prime, ...
                  x, y, 1];
end

% Perform Singular Value Decomposition (SVD) on A_12
fprintf('Performing SVD of matrix A for frame 1 and frame 2...\n');
[U, S, V] = svd(A_12);

% Extract the Fundamental Matrix (F) from the SVD
F_12 = reshape(V(:, end), 3, 3)'; % Last column of V reshaped to 3x3

% Normalize F
F_12 = F_12 / norm(F_12);

% Display the Fundamental Matrix
fprintf('Fundamental Matrix (F) for frame 1 and frame 2:\n');
disp(F_12);

% Generate the A matrix for frame 1 and frame 3
fprintf('Generating the A matrix for frame 1 and frame 3...\n');

n = size(matched_points1_13, 1); % Number of correspondences (Nc)
A_13 = zeros(n, 9); % Initialize A matrix for frame 1 and frame 3

for i = 1:n
    % Extract points from frame 1 and frame 3
    x = matched_points1_13(i, 1);
    y = matched_points1_13(i, 2);
    x_prime = matched_points3_13(i, 1);
    y_prime = matched_points3_13(i, 2);

    % Populate the A matrix
    A_13(i, :) = [x_prime * x, x_prime * y, x_prime, ...
                  y_prime * x, y_prime * y, y_prime, ...
                  x, y, 1];
end

% Perform Singular Value Decomposition (SVD) on A_13
fprintf('Performing SVD of matrix A for frame 1 and frame 3...\n');
[U, S, V] = svd(A_13);

% Extract the Fundamental Matrix (F) from the SVD
F_13 = reshape(V(:, end), 3, 3)'; % Last column of V reshaped to 3x3

% Normalize F
F_13 = F_13 / norm(F_13);

% Display the Fundamental Matrix
fprintf('Fundamental Matrix (F) for frame 1 and frame 3:\n');
disp(F_13);





% COMPUTE ESSENTIAL MATRIX FROM FUNDAMENTAL MATRIX

% Compute Essential Matrix (E) from Fundamental Matrix (F)
fprintf('Computing Essential Matrix (E)...\n');
E = K' * F * K;
disp('Essential Matrix (E):');
disp(E);







% DECOMPOSE ESSENTIAL MATRIX INTO R AND t

% Decompose Essential Matrix into Rotation (R) and Translation (t)
[U, S, V] = svd(E);
if det(U * V') < 0
    V = -V;
end

% Two possible rotations
R1 = U * [0 -1 0; 1 0 0; 0 0 1] * V';
R2 = U * [0 1 0; -1 0 0; 0 0 1] * V';

% Translation vector (baseline direction)
t = U(:, 3);

% Display results
disp('Possible Rotation Matrix R1:');
disp(R1);
disp('Possible Rotation Matrix R2:');
disp(R2);
disp('Translation Vector (t):');
disp(t);

% VALIDATION (OPTIONAL)
% Ensure that R1 and R2 are valid rotation matrices
if det(R1) < 0
    fprintf('R1 determinant is negative, flipping sign.\n');
    R1 = -R1;
end

if det(R2) < 0
    fprintf('R2 determinant is negative, flipping sign.\n');
    R2 = -R2;
end



% DEPTH ESTIMATION

fprintf('Calculating depth maps with calibrated parameters...\n');

% Compute baseline length from translation vector
baseline = norm(t);

% Calculate depth maps using the dense disparity maps
depth_map_12 = (K(1,1) * baseline) ./ dense_disparity_12; % focal_length = K(1,1)
depth_map_13 = (K(1,1) * baseline) ./ dense_disparity_13;

% Replace invalid values in depth maps
depth_map_12(isinf(depth_map_12) | isnan(depth_map_12)) = 0;
depth_map_13(isinf(depth_map_13) | isnan(depth_map_13)) = 0;

% Plot depth maps
figure;
subplot(1, 2, 1);
imagesc(depth_map_12);
title('Depth Map (Frame 1 - Frame 2)');
colorbar;
colormap jet;

subplot(1, 2, 2);
imagesc(depth_map_13);
title('Depth Map (Frame 1 - Frame 3)');
colorbar;
colormap jet;




% 3D RECONSTRUCTION

fprintf('Reconstructing 3D point clouds with calibrated parameters...\n');

% Generate 3D point clouds
[y_coords, x_coords] = meshgrid(1:w, 1:h);
z_flat_12 = depth_map_12(:);
z_flat_13 = depth_map_13(:);

% Compute 3D points for each depth map
points_3d_12 = [x_coords(:) .* z_flat_12 / K(1,1), ...
                y_coords(:) .* z_flat_12 / K(2,2), ...
                z_flat_12];

points_3d_13 = [x_coords(:) .* z_flat_13 / K(1,1), ...
                y_coords(:) .* z_flat_13 / K(2,2), ...
                z_flat_13];

% Remove invalid points
points_3d_12 = points_3d_12(z_flat_12 > 0, :);
points_3d_13 = points_3d_13(z_flat_13 > 0, :);

% Save 3D point clouds
save('points_3d_12.mat', 'points_3d_12');
save('points_3d_13.mat', 'points_3d_13');




% VISUALIZE THE POINT CLOUD

fprintf('Visualizing reconstructed 3D point clouds...\n');

figure;
scatter3(points_3d_12(:, 1), points_3d_12(:, 2), points_3d_12(:, 3), 1, points_3d_12(:, 3), 'filled');
title('3D Point Cloud (Frame 1 - Frame 2)');
xlabel('X');
ylabel('Y');
zlabel('Z');
colormap jet;
colorbar;

figure;
scatter3(points_3d_13(:, 1), points_3d_13(:, 2), points_3d_13(:, 3), 1, points_3d_13(:, 3), 'filled');
title('3D Point Cloud (Frame 1 - Frame 3)');
xlabel('X');
ylabel('Y');
zlabel('Z');
colormap jet;
colorbar;

fprintf('3D reconstruction and visualization completed.\n');