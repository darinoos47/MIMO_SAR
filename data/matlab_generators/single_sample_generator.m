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
angle_steps = 1001;
angle_resolution = (start_angle - end_angle)/(angle_steps-1);
A = zeros(virtual_antenna_number,angle_steps); % steering matrix
theta = start_angle:-angle_resolution:end_angle;
for i=1:angle_steps
temp = exp(1i*2*pi*virtual_antenna_spacing/wavelength*(0:virtual_antenna_number-1)*sind(theta(i)));
A(:,i)= temp.';
end
A = A/sqrt(1001);

%% Forward Operator
angles = linspace(start_angle, end_angle, angle_steps);
angles = angles(:);
x = zeros(size(angles));
x(501) = 1;
y = A*x;
x = x.';
% Save data for Python

fprintf('Saving data for Python...\n');
% save('FL_MIMO_SAR_data.mat', 'A', 'received_signals_fft', 'object', '-v7.3');

received_signals_fft = y.'; % For debugging purpose. Only keep on range sample.
save('FL_MIMO_SAR_data.mat', 'A', 'received_signals_fft', 'x');
fprintf('Data saved to FL_MIMO_SAR_data.mat\n');