function [features, weights,gtDensities] = extract_features(input_folder,image,Dict)

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

im_path=[input_folder image];
im = imread(im_path);
grayIm=rgb2gray(im); %image in grey scale
[f d] = vl_dsift(single(grayIm)); %computing the dense sift descriptors centered at each pixel
%estimating the crop parameters where SIFTs were not computed:
% f=FRAMES is a 2 x NUMKEYPOINTS, each colum storing the center (X,Y)
% of a keypoint frame (all frames have the same scale and orientation).
% d= DESCRS is a 128 x NUMKEYPOINTS matrix with one
% descriptor per column, in the same format of VL_SIFT()
minf = floor(min(f,[],2)); %[Y,I] = MIN(X,[],DIM) operates along the dimension DIM.
maxf = floor(max(f,[],2));
minx = minf(1);
miny = minf(2);
maxx = maxf(1);
maxy = maxf(2);
%compute features
features = vl_ikmeanspush(uint8(d),Dict);
features = reshape(features, maxy-miny+1, maxx-minx+1);
weights = ones(size(features));
%computing ground truth densities:
[~,name,~] = fileparts(im_path);
ground_truth = imread([input_folder name 'dots.png']);
gtDensities = ground_truth;
gtDensities = double(gtDensities(:,:,1))/255; %using the red channel
gtDensities = imfilter(gtDensities,fspecial('gaussian', 15.0*6,15));
gtDensities = gtDensities(miny:maxy,minx:maxx); %cropping GT densities to match the window where features are computable


end
