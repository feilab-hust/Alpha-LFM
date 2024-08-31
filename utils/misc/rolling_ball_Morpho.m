function [ background ] = rolling_ball_Morpho(image,R,H)
%rolling_ball(image, R, H) takes a single image and calculates a rolling
%ball background, with the necessary inputs as the image matrix, the radius
%and the height of the ball.  (Default R=15, H=5).
%   Prior to the morphological opening using a rolling ball, there is a
%   conservative disk averaging filter of 2-pixel radius to smooth the
%   background.  Presumably, any non-background objects biased by the
%   averaging filter are removed by the morphological opening.

% Copyright 2015 Danny Crawford, Brandeis University.

% This is free software: you can redistribute it and/or modify it under the
% terms of the GNU General Public License as published by the Free Software
% Foundation, either version 3 of the License, or (at your option) any later
% version.

% This software is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
% A PARTICULAR PURPOSE. See the GNU General Public License for more details.
% You should have received a copy of the GNU General Public License
% along with this software. If not, see <http://www.gnu.org/licenses/>.


% the first process is a small disk background averaging filter, to smooth the
% baseline noise
h=fspecial('disk',2);
%I0=imfilter(image,h,'symmetric','same');
I0=image;
%setup default R and H values
if (nargin < 3) || isempty(H)
  H = 5;
end
if (nargin < 3) || isempty(R)
  R = 15;
end
% the structural element of type ball takes the inputs of 'r' and 'h'.
se=strel('ball', R, H);

% now morphologically open the averaged image, using the structural element
% se

background=imopen(I0, se);

end
