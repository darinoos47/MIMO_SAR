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
num_training_samples = 200; % Create 5000 (x, y) pairs
noise_std = 1e-20;            % Standard deviation of the complex noise

% <<< NEW: Parameters for diversified targets >>>
max_num_targets = 4;       % Max targets per sample (will randomly pick 1, 2, 3, or 4)
target_width_options = [1, 3, 5]; % Possible target widths (in bins)

fprintf('Generating %d training samples...\n', num_training_samples);
fprintf('  Max targets per sample: %d\n', max_num_targets);
fprintf('  Target width options: [%s]\n', num2str(target_width_options));

% --- Initialize Dataset Matrices ---
x_dataset = zeros(num_training_samples, angle_steps);
y_dataset = zeros(num_training_samples, virtual_antenna_number);

% --- Generate data using a parallel loop ---
parfor i = 1:num_training_samples
    % Create an empty ground truth 'x' for this sample
    x_sample_col = zeros(angle_steps, 1);
    
    % Randomly decide how many targets to add in this sample
    num_targets = randi(max_num_targets);
    
    for j = 1:num_targets
        % --- Pick a random width ---
        width_idx = randi(length(target_width_options));
        target_width = target_width_options(width_idx);
        half_width = floor(target_width / 2);
        
        % --- Pick a random center ---
        center_index = randi(angle_steps);
        
        % Determine start and end indices, handling edges
        start_index = max(1, center_index - half_width);
        end_index = min(angle_steps, center_index + half_width);
        
        % Add this target to the 'x' sample
        % (using max ensures we don't erase overlapping targets, just join them)
        x_sample_col(start_index:end_index) = max(x_sample_col(start_index:end_index), 1.0);
    end
    
    % Generate the clean measurement 'y'
    y_sample_col = A * x_sample_col;
    
    % Add complex Gaussian noise
    noise = (randn(size(y_sample_col)) + 1i * randn(size(y_sample_col))) * noise_std / sqrt(2);
    y_sample_noisy_col = y_sample_col + noise; % Noise is ON
    
    % Store the ROW vectors in our dataset
    x_dataset(i, :) = x_sample_col.';
    y_dataset(i, :) = y_sample_noisy_col.';
end

fprintf('Dataset generation complete.\n');

%% Save data for Python
fprintf('Saving data for Python...\n');

% Save the full datasets
% 'x' is now [5000, 51]
% 'received_signals_fft' is now [5000, 8]
x = x_dataset;
received_signals_fft = y_dataset;

% Use -v7.3 flag to handle large dataset files
save('FL_MIMO_SAR_data.mat', 'A', 'received_signals_fft', 'x', 'noise_std', '-v7.3');
fprintf('Data saved to FL_MIMO_SAR_data.mat\n');