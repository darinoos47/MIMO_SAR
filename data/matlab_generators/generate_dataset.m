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
fprintf('Generating %d training samples...\n', angle_steps);

% --- Define Dataset Parameters ---
num_training_samples = 100; % We will create 2000 (x, y) pairs
noise_std = 1e-3; % Standard deviation of the complex noise

% --- Initialize Dataset Matrices ---
% Each ROW will be one sample
% x_dataset: [num_training_samples, angle_steps]
% y_dataset: [num_training_samples, virtual_antenna_number]
x_dataset = zeros(num_training_samples, angle_steps);
y_dataset = zeros(num_training_samples, virtual_antenna_number);

% --- Create random target locations ---
% A list of random indices, one for each sample
target_indices = randi(angle_steps, [num_training_samples, 1]);

% --- Generate data using a parallel loop ---
for i = 1:num_training_samples
    % Create a single sparse target (ground truth 'x')
    x_sample_col = zeros(angle_steps, 1);
    x_sample_col(target_indices(i)) = 1.0; % Target at a random angle
    
    % Generate the clean measurement 'y'
    y_sample_col = A * x_sample_col;
    
    % Add complex Gaussian noise
    % (randn + 1i*randn)/sqrt(2) has a standard deviation of 1
    noise = (randn(size(y_sample_col)) + 1i * randn(size(y_sample_col))) * noise_std / sqrt(2);
    y_sample_noisy_col = y_sample_col + 0*noise;
    
    % Store the ROW vectors in our dataset
    x_dataset(i, :) = x_sample_col.';
    y_dataset(i, :) = y_sample_noisy_col.';
end

fprintf('Dataset generation complete.\n');

%% Save data for Python
fprintf('Saving data for Python...\n');

% Save the full datasets
% 'x' is now [2000, 1001]
% 'received_signals_fft' is now [2000, 8]
x = x_dataset;
received_signals_fft = y_dataset;

% Use -v7.3 flag to handle large dataset files
save('FL_MIMO_SAR_data.mat', 'A', 'received_signals_fft', 'x', 'noise_std', '-v7.3');
fprintf('Data saved to FL_MIMO_SAR_data.mat\n');