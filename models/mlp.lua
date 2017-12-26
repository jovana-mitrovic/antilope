require 'nn'
require 'utils.aux_fct'

--------------------
---- RELU LAYER ----
--------------------
function relu_layer(opt, inputSize, outputSize, drop_rate)
    local m = nn.Sequential()
    m:add(nn.Linear(inputSize, outputSize))
    if opt.batchNorm then m:add(nn.BatchNormalization(outputSize)) end
    m:add(nn.ReLU())
    if opt.dropout then m:add(nn.Dropout(drop_rate)) end
    return m
end




---------------------
---- BUILD MODEL ----
---------------------
-- function build_mlp(...)

    model = nn.Sequential()
    -- if opt.dataset == 'cifar10' then
    --    dofile '/homes/mitrovic/Desktop/torch/cifar10/batchFlip.lua'
    --    model:add(nn.BatchFlip():float())
    -- end
    -- model:add(cast(nn.Copy('torch.DoubleTensor', torch.type(cast(torch.Tensor())))))
    local sizes = torch.Tensor(torch.totable(train_data.data:size()))
    local prevSize = sizes:index(1, torch.range(2, sizes:size(1)):long())
    prevSize = prevSize:cumprod()[prevSize:size(1)]
    model:add(cast(nn.View(prevSize)))

    -- Hidden Layers
    for i, nextSize in ipairs(opt.nHiddenUnits) do
        if i == #opt.nHiddenUnits then drop_rate = 0.3  else  drop_rate = 0.5 end
        model:add(relu_layer(opt, prevSize, nextSize, drop_rate))
        prevSize = nextSize
    end

    model:add(nn.Linear(prevSize, #classes))


    -- return model
-- end

    -- -- Output layer
    -- if list_weight_param[#list_weight_param].type == 'KLinear' then
    --     require 'core.layers.KLinear'
    --     model:add(nn.KLinear(nextSize, #classes, assignments[#assignments], opt.KLinearParam))
    -- else
    --     model:add(nn.Linear(prevSize, #classes))
    -- end

-- Dropout and Batch Normalization at output layer
-- if opt.batchNorm then model:add(nn.BatchNormalization(#classes)) end
-- if opt.dropout then model:add(nn.Dropout(drop_rate)) end


return model
