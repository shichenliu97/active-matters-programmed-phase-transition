%% PIV and Correlation Analysis

clc; clear; close all;

saveimg     = false;

sample      = '235_MT_correlation_15ul';
vidname     = strcat(sample , '.avi');
filename    = strcat('testcases/',vidname);
vid         = VideoReader(filename);
im1         = read(vid,100);

EdgeLength  = 40;   % Length of box edges in pixels; 
smooth      = 1;    % set to 1 if gaussian smoothing is desired
smooth_len  = 10;

pxlen   = .43 ;
xrange  = 1:size(im1 , 2);
yrange  = 851:1050;
Ttot    = vid.NumFrames;

im1 = rescale(im1(yrange , xrange));

tmin    = 10;
tmax    = 100;
delta_T = 1;  %PIV step      
step    = 5;  %step in timeframes.
tseries = tmin:step:tmax;

vscale  = 1;

[X1,Y1] = meshgrid(EdgeLength/2:EdgeLength:length(xrange)-EdgeLength/2,...
                   EdgeLength/2:EdgeLength:length(yrange)-EdgeLength/2); 

corr_len = zeros(length(tseries),1);

tp = 0;
for tt = tseries
    tp = tp + 1;
    im1 = read(vid,tt);
    im2 = read(vid,tt+delta_T);

    im1 = rescale(im1(yrange , xrange)); %intensity normalization
    im2 = rescale(im2(yrange , xrange));

    [VV , moved] = imregdemons(im1 , im2, [100,50,25] );

    VX = - squeeze(VV(:,:,1))/delta_T;
    VY = - squeeze(VV(:,:,2))/delta_T;

    if smooth
        VX = imgaussfilt(VX , smooth_len);
        VY = imgaussfilt(VY , smooth_len);
    end

    VX = VX - mean(VX(:));
    VY = VY - mean(VY(:));

    PX = im1.*VX;
    PY = im1.*VY;

    UX = vscale * imresize(PX , [size(Y1 , 1) , size(X1 , 2)]);
    UY = vscale * imresize(PY , [size(Y1 , 1) , size(X1 , 2)]);

    PXmean = mean(PX,1);
    PXsmth = smoothdata(PXmean,'gaussian');
    PXsmth = PXsmth/max(abs(PXsmth));

    PXprod = PXsmth(1:end-1) .* PXsmth(2:end);
    PXzc = find(PXprod<0);

    PXd1 = diff(PXsmth,1);

    PXsink = zeros(1,1);
    nsink  = 0;
    for ii = PXzc
        if PXd1(ii)<0
            nsink = nsink+1;
            PXsink(nsink) = ii;
        end
    end

    PXsrc = setdiff(PXzc , PXsink);
    PXsource = zeros(length(PXsrc)+2,1);
    PXsource(1) = 1;
    PXsource(end) = max(xrange);
    PXsource(2:end-1) = PXsrc;
    PXsrc = PXsource;

    peaks = zeros(1,1);
    wells = zeros(1,1);
    pn = 0;
    for ss = 1:length(PXsrc)-1
        pn = pn + 1;
        [valp , peaks(pn)] = max(PXsmth(PXsrc(ss):PXsrc(ss+1)));
        [valw , wells(pn)] = min(PXsmth(PXsrc(ss):PXsrc(ss+1)));
    end

    gaps = wells - peaks;
    gaps = gaps(gaps>0);
    corr_len(tp) = pxlen * mean(gaps);
    imz = im1(floor(length(yrange)/2),:);
    imzsmooth = smoothdata(imz,'gaussian');
    plot(imzsmooth);

end

figure ; 
plot(tseries,corr_len,'LineWidth',5);
xlabel('Time');
ylabel('Correlation Length / um');
saveas(gcf,'CorrLength.png');
saveas(gcf,'CorrLength');


