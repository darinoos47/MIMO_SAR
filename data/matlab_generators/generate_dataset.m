%% Cleaning
clear all;
close all;
clc;
set(0, 'DefaultFigureRenderer', 'painters');

%% Parameters
% checked
carrier_frequency = 77e9;
wavelength = 3e8/carrier_frequency;
virtual_antenna_spacing = 0.5*wavelength;
virtual_antenna_number = 8;
y_virtual_array = linspace(3.5*virtual_antenna_spacing,-3.5*virtual_antenna_spacing,virtual_antenna_number);
bandwidth = 2e9;
pulse_duration = 200e-6;
chirp_rate = bandwidth/pulse_duration;
samling_rate = 10e6;
number_of_samples = round(pulse_duration*samling_rate);
range_resolution = 3e8/(2*bandwidth);

%% Steering Matrix
start_angle = 25;
end_angle = -25;
angle_steps = 51;
angle_resolution = (start_angle - end_angle)/(angle_steps-1);
A = zeros(virtual_antenna_number,angle_steps); % steering matrix
theta = start_angle:-angle_resolution:end_angle;
for i=1:angle_steps
    temp = exp(1i*2*pi*virtual_antenna_spacing/wavelength*(0:virtual_antenna_number-1)*sind(theta(i)));
    A(:,i)= temp.';
end
% Normalize the steering matrix
A = A/sqrt(angle_steps);

%% Generate Multi-Sample Dataset

% --- Define Dataset Parameters ---
% <<< MODIFICATION #1: Target width in bins (use an odd number) >>>
target_width_bins = 1; % Set to 1 for sparse, 3 or 5 for wider targets

% <<< MODIFICATION #2: Loop deterministically for each angle bin >>>
num_training_samples = angle_steps; % Generate one sample for each angle
half_width = floor(target_width_bins / 2);

noise_std = 1e-20; % Standard deviation of the complex noise
fprintf('Generating %d training samples with target width %d...\n', num_training_samples, target_width_bins);

% --- Initialize Dataset Matrices ---
x_dataset = zeros(num_training_samples, angle_steps);
y_dataset = zeros(num_training_samples, virtual_antenna_number);

% --- Generate data using a deterministic loop ---
for i = 1:num_training_samples
    % Create a single sparse target (ground truth 'x')
    x_sample_col = zeros(angle_steps, 1);
    
    % <<< MODIFICATION #3: Create the wider target with boundary checks >>>
    start_index = max(1, i - half_width);
    end_index = min(angle_steps, i + half_width);
    x_sample_col(start_index:end_index) = 1.0; % Target centered at 'i'
    
    % Generate the clean measurement 'y'
    y_sample_col = A * x_sample_col;
    
    % Add complex Gaussian noise
    noise = (randn(size(y_sample_col)) + 1i * randn(size(y_sample_col))) * noise_std / sqrt(2);
    y_sample_noisy_col = y_sample_col + 0*noise; % Noise is still turned off here
    
    % Store the ROW vectors in our dataset
    x_dataset(i, :) = x_sample_col.';
    y_dataset(i, :) = y_sample_noisy_col.';
end

fprintf('Dataset generation complete.\n');

%% Save data for Python
fprintf('Saving data for Python...\n');

% Save the full datasets
% 'x' is now [51, 51]
% 'received_signals_fft' is now [51, 8]
x = x_dataset;
received_signals_fft = y_dataset;

% Use -v7.3 flag to handle large dataset files
save('../FL_MIMO_SAR_data.mat', 'A', 'received_signals_fft', 'x', 'noise_std', '-v7.3');
fprintf('Data saved to FL_MIMO_SAR_data.mat\n');

%% Save object for image reconstruction
x_domain_length = 4.125;
y_domain_length = 4.125;
dx = range_resolution;
dy = range_resolution;
x = (dx/2):dx:(x_domain_length-dx/2);
y = (y_domain_length/2-dy/2):-dy:(-y_domain_length/2+dy/2);
[x_grid,y_grid] = meshgrid(x,y);


ranges = linspace(5, 9.3552, 51);
ranges = ranges(:);
angles = theta(:);
angles = flipud(angles);
x_radar = -5;
y_radar = 0;
save('../data.mat', 'A', 'received_signals_fft', 'x', 'angles', 'ranges', 'x_grid', 'y_grid', 'x_radar', 'y_radar', '-v7.3');



dx = x_grid - x_radar;
dy = y_grid - y_radar;

r = sqrt(dx.^2 + dy.^2);
theta_deg = atan2(dy, dx) * 180/pi ;



