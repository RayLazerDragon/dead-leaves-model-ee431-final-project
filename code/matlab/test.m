% options.shape = 'disk'; 
% options.nbr_iter = 1000;
% M = compute_dead_leaves_image(500,3,options);
% 
% imshow(M)
shape = 'disk'
nbr_iter = 5000;
sigma = 3
rmin=0.01;
rmax=1;

n=500;

M = zeros(500)+Inf;
x = linspace(0,1,n);

[Y,X] = meshgrid(x,x);

k = 200;
      % sampling rate of the distrib
r_list = linspace(rmin,rmax,k);
r_dist = 1./r_list.^sigma;

% if sigma>0
%     r_dist = r_dist - 1/rmax^sigma;
% end

% r_dist = rescale( cumsum(r_dist) );
% m = n^2;
% 
% for i=1:nbr_iter
%     
% 
% 	% compute scaling using inverse mapping
%     r = rand(1);
%     [tmp,I] = min( abs(r-r_dist) );
%     r = r_list(I);
%     
%     x = rand(1);    % position 
%     y = rand(1);
%     a = rand(1);    % albedo
%     
%     isinf(M) & (X-x).^2 + (Y-y).^2<r^2;
%     
%     if strcmp(shape, 'disk')
%         I = find(isinf(M) & (X-x).^2 + (Y-y).^2<r^2 );
%     else
%         I = find(isinf(M) & abs(X-x)<r & abs(Y-y)<r );
%     end
%     
%     break;
% end