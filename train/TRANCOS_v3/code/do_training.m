function do_training(input_folder,output_folder,training_im_list)
% do_training Train the system.
%
%
% Arguments:
%   -input_folder: input image folder.
%   -output_folder: folder where to store the generated files.
%   -training_im_list: text file which contains all the image names to
%   process.

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

%Here we show how to train the model of Lempitsky and Zisserman used in our
%paper. This way anyone can reproduce our results using the TRANCOS
%dataset.

%We first load a pre-trained SIFT codebook obtained with the TRANCOS
%training images.
disp('Loading pre-trained SIFT codebook...');
dictionary_path = ['Lempitsky', filesep, 'dictionary2000.mat'];
load(dictionary_path,'Dict')
nfeatures = size(Dict,2);
D = Dict;

%% Read training images
images=textread(training_im_list,'%s\n');
num_images=size(images,1);

%Allocate memory for features, weights and densities
features = cell(num_images,1);
weights = cell(num_images,1);
gtDensities = cell(num_images,1);


%% Compute training features
feat_out_folder=[output_folder, 'features_training.mat'];

if exist(feat_out_folder)
    disp('Loading features precomputed at previous run');
    load(feat_out_folder,'features','weights','gtDensities');
else
    parfor j=1:num_images
        disp(['Processing image #' num2str(j) ' (out of ' num2str(num_images) ')...']);         
        [features{j},weights{j},gtDensities{j}] = extract_features(input_folder,images{j},D);        
    end
    save(feat_out_folder,'features','weights','gtDensities');
end



%% Training the model
disp('Training the model');

weights_folder=[output_folder, 'model.mat'];

if exist(weights_folder)
    disp('The model was learned before. We load its weights.');
else
    fprintf('Now using the %d images to train the model.\n',num_images);    
    maxIter = 10000;
    weightMap = ones([size(features{1},1) size(features{1},2)]);
    C = 1000; % Penalty value to train the regressor.

    disp('--------------');
    disp('Training the model with L1 regularization:');
    disp('--------------');
    wL1 = learn_to_count(nfeatures, features, weights, ...
            weightMap, gtDensities, -C/(num_images), maxIter);
    save(weights_folder,'wL1');
end
