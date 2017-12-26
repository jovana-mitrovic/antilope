---------------------------------------------------------
----- [[ command line arguments for mnist dataset]] -----
---------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('--save', 'logs/', 'subdirectory to save logs')
cmd:option('--end_path', '', 'end of path where to save file')
cmd:option('--backend', 'nn')
cmd:option('--type', 'float', 'cuda/float/cl')

-- [[ mnist ]] --
cmd:option('--trainSize', 60000, 'size of training set')
cmd:option('--testSize', 10000, 'size of testing set')

-- [[ model ]] --
cmd:option('--model', 'mlp', 'possible models: mlp | vgg_bn_drop | nin')
cmd:option('--nHiddenUnits', {800}) --{2500,2000,1500,1000,500}, 'number of hidden units per layer')
cmd:option('--classHiddenUnits', {800})

cmd:option('--dropout', false)
cmd:option('--batchNorm', false)

cmd:option('--input_rescale', false, 'rescale input to [0,1]')
cmd:option('--proj_rescale', false, 'rescale projection points to [0,1]')
-- cmd:option('--input_zca', false, 'apply ZCA Whitening on input')
-- cmd:option('--proj_zca', false, 'apply ZCA Whitening on projection points')
-- cmd:option('--input_lecunlcn', false, 'apply Local Contrast Normalization on input')
-- cmd:option('--proj_lecunlcn', false, 'apply Local Contrast Normalization on projection points')

cmd:option('--weightInit', 'kaiming') --, 'kernel', 'kernel', 'kernel','kernel', 'KLinear'} , 'type of weight initilization')
cmd:option('--weightScale', 0.01)
cmd:option('--numPos', 10, 'number of positive examples per neuron')
cmd:option('--numNeg', 0, 'number of negative examples per neuron per class')
cmd:option('--distribution', 'uniform', 'distribution of alphas for kernel layer: normal, uniform')

cmd:option('--numPoints', {20000}, 'number of random points for Fisher LDA')

cmd:option('--shuffle', false, 'shuffle schedule of projection points')
cmd:option('--KLinearParam', -0.05)


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

cmd:option('--maxEpoch', 10)
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
