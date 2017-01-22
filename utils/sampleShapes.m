function SS = sampleShapes(S,ind)

SS = [];
for j = ind
    SS = [SS;S(j*3-2:j*3,:)];
end