% This script reproduces the results described in our paper using the
% method of Lempitsky and Zisserman, introduced in the following
% publication:
% V. Lempitsky and A. Zisserman. Learning to Count Objects in Images. NIPS, 2010.
% This script can be used to learn how to work with the TRANCOS dataset in
% order to create novel counting solutions.
% Have fun!

%AUTORIGHTS
%Copyright (C) 2015 Roberto Javier López-Sastre
%
%This file is part of TRANCOS-Software
%
%TRANCOS-Software is free software; you can redistribute it and/or modify
%it under the terms of the GNU General Public License as published by
%the Free Software Foundation; either version 2, or (at your option)
%any later version.
%
%This program is distributed in the hope that it will be useful,
%but WITHOUT ANY WARRANTY; without even the implied warranty of
%MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%GNU General Public License for more details.
%
%You should have received a copy of the GNU General Public License
%along with this program; if not, write to the Free Software Foundation,
%Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
%

%% Define paths

% path to save final and intermediate results.
results_path = ['..', filesep, 'results', filesep];

% Create results directory
if ~exist(results_path)
    mkdir(results_path);
end

% images path
input_folder= ['..', filesep, 'images', filesep];

%define image set to use during training and testing
image_set = 'trainval'; %training, validation, trainval
training_im_list=['..', filesep, 'image_sets', filesep, image_set, '.txt'];

test_im_list=['..', filesep, 'image_sets', filesep 'test.txt'];



%% Training a new model
% Call do_training to learn your model for object counting.
do_training(input_folder,results_path,training_im_list);

%% Testing your model
% Call do_test to run your model using the test images
do_test(input_folder,results_path,test_im_list);


%% Evaluate the model
% Here we provide the code to evalaute the model. We show how to compute
% the GAME metric in order to report results in the TRANCOS dataset.

%The do_test script saves the densitites estimated for each test image in
%the folder output_densities, within the results_path. We simply have to
%read these densities and compare them with the GT, using the GAME metric.

densities_folder = [results_path,  'output_densities', filesep];
files = dir([densities_folder '*.mat']);
num_files = size(files,1);

%Game metric level (L)
game_level = 3;  %0,1,2,3

%Allocating memory
single_game = zeros(game_level,num_files);
mean_game = zeros(1, game_level);

for l = 1:game_level+1
    for a = 1:num_files
        load([densities_folder, '/', files(a).name]);
        single_game(l,a) = GAME_recursive(estDensity_ROI, gt, 0, l-1);
    end    
    mean_game(l) = mean(single_game(l,:));
    fprintf('Mean GAME %i: %f\n', l-1, mean_game(l));
end

save([results_path 'GAME-results.mat'], 'mean_game');

% Plot results
fHand = figure;
aHand = axes('parent', fHand);
hold(aHand, 'on')
colors = hsv(numel(mean_game));
for i = 1:numel(mean_game)
    bar(i-1, mean_game(i), 'parent', aHand, 'facecolor', colors(i,:));
end

title('GAME Error Measurement')
xlabel('GAME Level')
ylabel('GAME')
