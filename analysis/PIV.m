gridSize = 30;   % Length of box edges in pixels 
smooth   = 1;    % set to 1 if gaussian smoothing is desired
sigma    = 30;   % standard deviation of gaussian kernel
velScale = 1.6;  % µ/px

filename = '/Users/scliu/Downloads/cell_move_cy5.tif';  % images are assumed to come in stacks
im1      = imread(filename,1);
xrange   = 1:size(im1,2);      
yrange   = 1:size(im1,1);
Ttot     = length(imfinfo(filename));

delta_T  = 1;    % time difference between frames to measure velocity
step     = 1;    % time steps between frames at which velocity is measured
tseries  = 1:step:Ttot-1;

[X1,Y1]  = meshgrid(gridSize/2:gridSize:length(xrange)-gridSize/2,...
                    gridSize/2:gridSize:length(yrange)-gridSize/2); 

for tt = tseries

    im1 = imread(filename,tt);
    im2 = imread(filename,tt+delta_T);
    
    im1 = im1(xrange , yrange);
    
    if ndims(im1) == 3 || ndims(im2) == 3
        im1 = squeeze(mean(im1,3));
        im2 = squeeze(mean(im2,3));
    end
    
    [VV,~] = imregdemons(im1,im2,[100,50,25]);
    
    VX = - squeeze(VV(:,:,1));
    VY = - squeeze(VV(:,:,2));

    if smooth
        VX = imgaussfilt(VX,sigma)/delta_T;
        VY = imgaussfilt(VY,sigma)/delta_T;
    end
    
    UX = imresize(VX , [size(Y1,1) , size(X1,2)]) * velScale;
    UY = imresize(VY , [size(Y1,1) , size(X1,2)]) * velScale;

    imshow(im1/255)
    hold on 
    velField1 = quiver(X1,Y1,UX,UY,'-c','LineWidth',2, 'AutoScale','off',...
                        'AlignVertexCenters' , 'on' );
    set(gca,'Color','k')
    axis equal
    
    close all
    
end