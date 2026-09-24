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

%% Setup project paths
addpath(genpath('.'));

%% Generate mex files
disp('Compiling maxsubarray2D.cpp ...');
mex(['.',filesep, 'Lempitsky', filesep, 'maxsubarray2D.cpp'], '-outdir', ['.', filesep, 'Lempitsky', filesep])

% Remember to install the VLFeat library and run the vl_setup script first
% VLFeat can be downloaded from: http://www.vlfeat.org/
vl_setup

