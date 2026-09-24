function do_test(input_folder,output_folder,test_im_folder)
% do_test Test the system.
%
%
% Arguments:
%   -input_folder: input image folder.
%   -output_folder: folder where to store the generated files.
%   -test_im_folder: text file which contains all the image names to
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

%We first load a pre-trained SIFT codebook obtained with the TRANCOS
%training images.
disp('Loading pre-trained SIFT codebook...');
dictionary_path = ['Lempitsky', filesep, 'dictionary2000.mat'];
load(dictionary_path,'Dict')
nfeatures = size(Dict,2);
D = Dict;

%Load the learned model
weights_folder=[output_folder, 'model.mat'];
load(weights_folder,'wL1');

%% Read test images
images=textread(test_im_folder,'%s\n');
num_images=size(images,1);



%% Compute test features
features = cell(num_images,1);
weights = cell(num_images,1);
gtDensities = cell(num_images,1);

feat_out_folder=[output_folder, 'features_test.mat'];

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


%% Testing the model
disp('Testing the model');

results_out_file=[output_folder, 'results.mat'];

if ~exist([output_folder, filesep, 'output_densities'])
    mkdir([output_folder, filesep, 'output_densities']);
end

if exist(results_out_file)
    disp('The results already exist');
else    
    trueCount = zeros(1,num_images);
    model1Count = zeros(1,num_images);    

    fprintf('Now evaluating on the remaining %d images.\n',num_images);
    for j=1:num_images       
        %identify image
        im_path=[input_folder images{j}];
        [~,name,~]=fileparts(im_path);
        
        %load region of interest       
        mask_folder=[input_folder name 'mask.mat'];
        load(mask_folder, 'BW');      
        BW=double(BW(1:size(gtDensities{j},1),1:size(gtDensities{j},2)));
        
        %Apply the ROI
        gtDensities_ROI{j} = BW.*gtDensities{j};
        trueCount(j) = sum(gtDensities_ROI{j}(:));

        %estimating the densities w.r.t. the learned model
        estDensity = wL1(features{j}).*weights{j};
        %apply the ROI
        estDensity_ROI = BW.*estDensity;        
        model1Count(j) = sum(estDensity_ROI(:));

        fprintf('Image #%d (of %d): trueCount = %f, model1 predicted count = %f\n',...
            j, num_images, trueCount(j), model1Count(j));
        
        gt = gtDensities_ROI{j} ;
        save([output_folder 'output_densities', filesep, 'image' num2str(j-1,'%.3d') '.mat'] , 'estDensity_ROI', 'gt');        
    end

    L1_error=mean(abs(trueCount - model1Count));
    
    fprintf('Model 1 (L1) average error = %f,\n', ...
        mean(abs(trueCount - model1Count)));

   
    save(results_out_file,'trueCount','model1Count','L1_error');
end
