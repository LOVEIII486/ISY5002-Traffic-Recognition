function game = GAME_recursive(density, gt,currentLevel, targetLevel)
% GAME_recursive function computes recursively the GAME metric in the
% current level, slicing the images (4 slices each) until reach the
% targetLevel.
% It can use zero-padding when the input images are not even, but does not
% affect the metric.

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

if currentLevel == targetLevel
    
    game = abs(sum(density(:)) - sum(gt(:)));
    return
    
else
    
    %Zero-padding to make the image even if needed
    dim = size(density);

    if mod(dim(1),2) ~= 0
        density = padarray(density, [1,0], 'post');
        gt = padarray(gt, [1,0], 'post');
    end

    if mod(dim(2),2) ~= 0
        density = padarray(density, [0,1], 'post');
        gt = padarray(gt, [0,1], 'post');
    end
    
    dim = size(density);

    %Creating the four slices
    density_slice{1} = density(1:dim(1)/2, 1:dim(2)/2);
    density_slice{2} = density(1:dim(1)/2, dim(2)/2+1:end);
    density_slice{3} = density(dim(1)/2+1:end, 1:dim(2)/2);
    density_slice{4} = density(dim(1)/2+1:end, dim(2)/2+1:end);
    
    gt_slice{1} = gt(1:dim(1)/2, 1:dim(2)/2);
    gt_slice{2} = gt(1:dim(1)/2, dim(2)/2+1:end);
    gt_slice{3} = gt(dim(1)/2+1:end, 1:dim(2)/2);
    gt_slice{4} = gt(dim(1)/2+1:end, dim(2)/2+1:end);

    currentLevel = currentLevel +1;
    
    for a =1:4
        res(a) = GAME_recursive(density_slice{a}, gt_slice{a}, currentLevel, targetLevel);
    end

    game = sum(res);
    
end
