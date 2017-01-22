require 'paths'
paths.dofile('util.lua')
paths.dofile('img.lua')

--------------------------------------------------------------------------------
-- Initialization
--------------------------------------------------------------------------------

if arg[1] == 'demo' or arg[1] == 'predict-test' then
    -- Test set annotations do not have ground truth part locations, but provide
    -- information about the location and scale of people in each image.
    a = loadAnnotations('test')

elseif arg[1] == 'predict-valid' or arg[1] == 'eval' then
    -- Validation set annotations on the other hand, provide part locations,
    -- visibility information, normalization factors for final evaluation, etc.
    a = loadAnnotations('valid')

else
    print("Please use one of the following input arguments:")
    print("    demo - Generate and display results on a few demo images")
    print("    predict-valid - Generate predictions on the validation set (MPII images must be available in 'images' directory)")
    print("    predict-test - Generate predictions on the test set")
    print("    eval - Run basic evaluation on predictions from the validation set")
    return
end

m = torch.load('umich-stacked-hourglass.t7')   -- Load pre-trained model

if arg[1] == 'demo' then
    idxs = torch.Tensor({695, 3611, 2486, 7424, 10032, 5, 4829})
    -- If all the MPII images are available, use the following line to see a random sampling of images
    -- idxs = torch.randperm(a.nsamples):sub(1,10)
else
    idxs = torch.range(1,a.nsamples)
end

if arg[1] == 'eval' then
    nsamples = 0
else
    nsamples = idxs:nElement() 
    -- Displays a convenient progress bar
    xlua.progress(0,nsamples)
    preds = torch.Tensor(nsamples,16,2)
end

--------------------------------------------------------------------------------
-- Main loop
--------------------------------------------------------------------------------

for i = 1,nsamples do
    -- Set up input image
    local im = image.load('images/' .. a['images'][idxs[i]])
    local center = a['center'][idxs[i]]
    local scale = a['scale'][idxs[i]]
    local inp = crop(im, center, scale, 0, 256)

    -- Get network output
    local out = m:forward(inp:view(1,3,256,256):cuda())
    cutorch.synchronize()
    local hm = out[2][1]:float()
    hm[hm:lt(0)] = 0

    -- Get predictions (hm and img refer to the coordinate space)
    local preds_hm, preds_img = getPreds(hm, center, scale)
    preds[i]:copy(preds_img)

    xlua.progress(i,nsamples)

    local predFile = hdf5.open('heatmaps/' .. a['images'][idxs[i]] .. '.h5', 'w')
    predFile:write('pred', preds_img)
    predFile:write('heatmap', hm)
    predFile:close()

    -- Display the result
    if arg[1] == 'demo' then
        preds_hm:mul(4) -- Change to input scale
        local dispImg = drawOutput(inp, hm, preds_hm[1])
        w = image.display{image=dispImg,win=w}
        sys.sleep(3)
    end

    collectgarbage()
end

-- Save predictions
if arg[1] == 'predict-valid' then
    local predFile = hdf5.open('preds/valid-example.h5', 'w')
    predFile:write('preds', preds)
    predFile:close()
elseif arg[1] == 'predict-test' then
    local predFile = hdf5.open('preds/test.h5', 'w')
    predFile:write('preds', preds)
    predFile:close()
elseif arg[1] == 'demo' then
    w.window:close()
end

--------------------------------------------------------------------------------
-- Evaluation code
--------------------------------------------------------------------------------

if arg[1] == 'eval' then
    -- Calculate distances given each set of predictions
    local labels = {'valid-example','valid-ours'}
    local dists = {}
    for i = 1,#labels do
        local predFile = hdf5.open('preds/' .. labels[i] .. '.h5','r')
        local preds = predFile:read('preds'):all()
        table.insert(dists,calcDists(preds, a.part, a.normalize))
    end

    require 'gnuplot'
    gnuplot.raw('set bmargin 1')
    gnuplot.raw('set lmargin 3.2')
    gnuplot.raw('set rmargin 2')    
    gnuplot.raw('set multiplot layout 2,3 title "MPII Validation Set Performance (PCKh)"')
    gnuplot.raw('set xtics font ",6"')
    gnuplot.raw('set ytics font ",6"')
    displayPCK(dists, {9,10}, labels, 'Head')
    displayPCK(dists, {2,5}, labels, 'Knee')
    displayPCK(dists, {1,6}, labels, 'Ankle')
    gnuplot.raw('set tmargin 2.5')
    gnuplot.raw('set bmargin 1.5')
    displayPCK(dists, {13,14}, labels, 'Shoulder')
    displayPCK(dists, {12,15}, labels, 'Elbow')
    displayPCK(dists, {11,16}, labels, 'Wrist', true)
    gnuplot.raw('unset multiplot')
end
