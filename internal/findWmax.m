function W_max = findWmax(heatmap)

W_max = zeros(2*size(heatmap,4),size(heatmap,3));
for j = 1:size(W_max,1)/2
    for i = 1:size(heatmap,3)
        [u,v] = find(heatmap(:,:,i,j)==max(max(heatmap(:,:,i,j))),1);
        W_max(2*j-1:2*j,i) = [v;u];
    end
end