require 'utils.whitening'
require 'utils.normalization'

c = require 'trepl.colorize'

function load_data(opt)
    -- Loading data depending on experiment options

    print(c.blue '==>' ..' loading data')
    if opt.dataset == 'mnist' then
        mnist = require 'datasets.mnist.mnist'
        train, test = mnist.traindataset(), mnist.testdataset()

        train_data = {}; train_data.data = torch.Tensor(opt.trainSize, 28*28)
        train_data.labels = torch.Tensor(opt.trainSize)
        for i = 1, opt.trainSize do
            train_data.data[{i}] = train[i].x:float()
            train_data.labels[{i}] = train[i].y + 1
        end

        test_data = {}; test_data.data = torch.Tensor(opt.testSize, 28*28)
        test_data.labels = torch.Tensor(opt.testSize)
        for i = 1, opt.testSize do
            test_data.data[{i}] = test[i].x:float()
            test_data.labels[{i}] = test[i].y + 1
        end

        -- added
        -- train_data = norm_normalization(train_data)
        -- test_data = norm_normalization(test_data)

        classes = {1,2,3,4,5,6,7,8,9,10}

    elseif opt.dataset == 'xor' then
        require 'datasets.xor.xor'; seed = 1
        dat = generate_xor(opt, seed)
        dat.data = pca_whiten(dat.data)
        dat = add_noise(dat, opt)
        dat, rot = rotation(dat)

        -- vectors spanning the plain where the signal is
        projmat = torch.Tensor(2, dat.data:size(2)):zero()
        projmat[{{1}, {1}}], projmat[{{2}, {2}}] = 1, 1
        projmat:addmm(projmat, rot:transpose(1,2))

        train_data, test_data = split_train_test(dat, 0.8)
        classes = {1,2}

    elseif opt.dataset == 'cifar10' then
        dataPath = '/data/greyostrich/oxwasp/oxwasp14/mitrovic/cifar.torch/'

        -- dofile  '/data/greyostrich/oxwasp/oxwasp14/mitrovic/cifar.torch/provider.lua'

        provider = torch.load(dataPath .. 'provider.t7')
        train_data, test_data = {}, {}

        train_data.data = provider.trainData.data[{{1,opt.trainSize}, {}, {}, {}}]:float()
        train_data.labels = provider.trainData.labels[{{1,opt.trainSize}}]

        test_data.data = provider.testData.data[{{1, opt.testSize}, {}, {}, {}}]
        test_data.labels = provider.testData.labels[{{1, opt.testSize}}]

        classes = {1,2,3,4,5,6,7,8,9,10}

   elseif opt.dataset == 'test_data' then
         require 'datasets.xor.xor'
         require 'datasets.test_data.test_data'; seed = 1
         dat = generate_test_data(opt, seed)

         train_data, test_data = split_train_test(dat, 0.8)
         classes = {1,2}

    else
       error('Unknown Dataset')
    end

    -- Rescaling the data
    if opt.input_rescale then
       train_data.data = train_data.data / torch.max(train_data.data)
       test_data.data = test_data.data / torch.max(test_data.data)
    end

    train_data.data = cast(train_data.data)
    test_data.data = cast(test_data.data)

    return train_data, test_data, classes, projmat
end


function load_projection_data(train_data)
     print(c.blue '==>' ..' setting projection points')

     projPoints = {}
     projPoints.data = train_data.data:clone():double()
     projPoints.labels = train_data.labels:clone()

     if opt.proj_rescale then
        projPoints.data = projPoints.data / torch.max(projPoints.data)
     end

     return projPoints
end


function flatten_propagated_data(data, model, outputSize)
       flat_prop_data = {}

       flat_prop_data.data = cast(torch.Tensor(data.data:size(1), outputSize)):zero()
       flat_prop_data.labels = data.labels:clone()

       for k = 1, data.data:size(1), 1000 do
           flat_prop_data.data[{{k, math.min(k + 1000, flat_prop_data.data:size(1))}}] =
              model:forward(data.data[{{k, math.min(k + 1000, flat_prop_data.data:size(1))}}])
       end

       return flat_prop_data
end
