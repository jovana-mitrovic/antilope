-------------------------------------------------------
----- [[ command line arguments for xor dataset]] -----
-------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('--save', 'logs/', 'subdirectory to save logs')
cmd:option('--end_path', 'allign_test/', 'end of path where to save file')
cmd:option('--backend', 'nn')
cmd:option('--type', 'float', 'cuda/float/cl')

-- [[ xor ]] --
cmd:option('--numClusters', 2, 'number of clusters for each class')
cmd:option('--numClusterPoints', 5000, 'number of points in each cluster')
cmd:option('--overlap', 0.5, 'extent of overlap')
cmd:option('--plot', false, 'plot xor data')
cmd:option('--nDim', 998, 'number of noise dimensions')
cmd:option('--noiseFactor', 0.01, 'standard deviation of noise in additional dimensions')

-- [[ model ]] --
cmd:option('--model', 'mlp', 'possible models: mlp | vgg_bn_drop | nin')
cmd:option('--nHiddenUnits', {10}, 'number of hidden units per layer')
cmd:option('--classifier', 'custom_init')
cmd:option('--classHiddenUnits', {10})
cmd:option('--dropout', false)
cmd:option('--batchNorm', true)

cmd:option('--input_rescale', false, 'rescale input to [0,1]')
cmd:option('--proj_rescale', false, 'rescale projection points to [0,1]')
-- cmd:option('--input_zca', false, 'apply ZCA Whitening on input')
-- cmd:option('--proj_zca', false, 'apply ZCA Whitening on projection points')
-- cmd:option('--input_lecunlcn', false, 'apply Local Contrast Normalization on input')
-- cmd:option('--proj_lecunlcn', false, 'apply Local Contrast Normalization on projection points')

cmd:option('--weightInit', 'kernel', 'type of weight initilization')

cmd:option('--weightScale', 100)
cmd:option('--numPos', 100, 'number of positive examples per neuron')
cmd:option('--numNeg', 10, 'number of negative examples per neuron per class')
cmd:option('--distribution', 'normal', 'distribution of alphas for kernel layer: normal, uniform')


cmd:option('--numPoints', {500}, 'number of random points for Fisher LDA')

cmd:option('--shuffle', false, 'shuffle schedule of projection points')
cmd:option('--KLinearParam', -0.01)


-- [[ optimization ]] --
cmd:option('--labelSmooth', true, 'regularization via label smoothing')

cmd:option('--optimization', 'adam', 'which optimization method to use: sgd | adagrad | adam')
cmd:option('--learningRate', 0.001, 'learning rate')
cmd:option('--lrDecay', 1e-7, 'learning rate decay factor')

cmd:option('--momentum', 0.9, 'SGD: momentum parameter')
cmd:option('--weightDecay', 0.0005, 'AdaGrad: weight decay rate')
cmd:option('--beta1', 0.9, 'Adam: first moment coefficient')
cmd:option('--beta2', 0.999, 'Adam: second moment coefficient')
cmd:option('--epsilon', 1e-8, 'Adam: for numerical stability')

cmd:option('--coefL1', 0, 'L1-regularization parameter')
cmd:option('--coefL2', 0, 'L2-regularization parameter')
cmd:option('--gradNoise', false, 'whether to add noise to the gradient')
cmd:option('--gradNoiseNu', 0.01, 'nu for gradNoise')
cmd:option('--gradNoiseGamma', 0.55, 'gamma for gradNoise')

cmd:option('--maxEpoch', 100)
cmd:option('--batchSize', 128)
cmd:option('--epoch_step', 5)
cmd:option('--numPatches', {10})
cmd:text()

opt = cmd:parse(arg or {})
package.path = package.path .. ';/Users/jovana/Desktop/nips_project/torch/core/?.lua'
require 'opt.test_opt'
test_cmd(opt)
-- opt.epoch_step = opt.maxEpoch

return opt
