-- Different weight initialization methods
-- > model = require('weight-init')(model, 'heuristic')

require 'nn'
require 'image'
require 'utils.aux_fct'
require 'core.layers.KernelLayer'
require 'core.layers.FisherLayer'
require 'utils.monitor'

-- "Efficient backprop", Yann Lecun, 1998
local function w_init_heuristic(fan_in, fan_out)
   return math.sqrt(1/(3*fan_in))
end

-- "Understanding the difficulty of training deep feedforward neural networks"
-- Xavier Glorot, 2010
local function w_init_xavier(fan_in, fan_out)
   return math.sqrt(2/(fan_in + fan_out))
end

local function w_init_xavier_caffe(fan_in, fan_out)
   return math.sqrt(1/fan_in)
end

-- "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification", Kaiming He, 2015
local function w_init_kaiming(fan_in, fan_out)
   return math.sqrt(4/(fan_in + fan_out))
end


local function w_init_simple(fan_in, fan_out)
    return 0.1
end


local function w_init_kernel(fan_in, fan_out, scale)
   return math.sqrt(1/(scale * fan_in))
end



function update_module(m)

     numAll = numAll + 1
     -- change when not only nn.linear will be updated
     if m.__typename == 'nn.Linear' then
        numLin = numLin + 1
        arg = list_weight_param[numLin].type
     else
        arg = list_weight_param[1].type
     end

     -- Choose weight init fct
     local fct = nil
     if     arg == 'heuristic'                  then fct = w_init_heuristic
     elseif arg == 'xavier'                     then fct = w_init_xavier
     elseif arg == 'xaviercaffe'                then fct = w_init_xavier_caffe
     elseif arg == 'kaiming'                    then fct = w_init_kaiming
     elseif arg == 'simple'                     then fct = w_init_simple
     elseif arg == 'kernel' or arg == 'fisher'  then fct = w_init_kaiming
     elseif arg == 'KLinear'                    then fct = w_init_kaiming
     else
        assert(false)
     end


     -- Update module
     if m.__typename == 'nn.SpatialConvolution' then
        m:reset(fct(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
     elseif m.__typename == 'nn.SpatialConvolutionMM' then
        m:reset(fct(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
     elseif m.__typename == 'nn.LateralConvolution' then
        m:reset(fct(m.nInputPlane*1*1, m.nOutputPlane*1*1))
     elseif m.__typename == 'nn.VerticalConvolution' then
        m:reset(fct(1*m.kH*m.kW, 1*m.kH*m.kW))
     elseif m.__typename == 'nn.HorizontalConvolution' then
        m:reset(fct(1*m.kH*m.kW, 1*m.kH*m.kW))
     elseif m.__typename == 'nn.TemporalConvolution' then
        m:reset(fct(m.weight:size(2), m.weight:size(1)))
     elseif m.__typename == 'nn.Linear' then
        -- use projection points if needed
        if arg == 'kernel' or arg == 'fisher' then

           -- projection points for each neuron
           local dat, labels = {}, {};
           for key, val in ipairs(wholeSchedule[numLin].indices) do
               dat[key] = projPoints.data:index(1, val:long()):double()
               labels[key] = projPoints.labels:index(1, val:long())
           end

           -- build model
           local prop_model = nn.Sequential()
           for key, val in pairs(flat_model:listModules()) do
              if key >= numAll then break end
              if val.__typename ~= 'nn.Sequential' then
                 prop_model:add(val:double())
              end
           end

           --propagate datapoints
           for key, val in pairs(dat) do
              dat[key] = prop_model:forward(dat[key])
           end

           -- update modules
           if arg == 'kernel' then
              m = nn.KernelLayer(m.weight:size(2), m.weight:size(1), dat, list_weight_param[numLin].scale)
           elseif arg == 'fisher' then
              m = nn.FisherLayer(m.weight:size(2), m.weight:size(1), {data = dat, labels = labels}, wholeSchedule[numLin])
           end
        elseif arg == 'KLinear' then
           m = nn.KLinear(m.weight:size(2), m.weight:size(1), assignments[numLin - 1], opt.KLinearParam)
        else
           m:reset(fct(m.weight:size(2), m.weight:size(1)))
        end

        -- set bias to zero
        m.bias:zero()
     end

     -- if propagate and m.__typename ~= 'nn.BatchFlip' and m.weight ~= nil then
     --    for i = 1, #new_features do
     --      new_features[i] = new_features[i]:typeAs(m.weight)
     --    end
     --    -- features = features:type(torch.type(m.weight)):clone()
     --
     --    -- [[ prevent memory overflow for complex models]]
     --    for i = 1, #new_features do
     --       local curr = m:forward(new_features[i]):clone()
     --       if curr:nDimension() == 1 then curr = curr:reshape(1, curr:size(1)) end
     --       if i == 1 then print(m.__typename, curr:size()) end
     --       new_features[i] = curr
     --    end

        -- new_features = deepcopy(newnew_features)
        -- newnew_features = nil
     collectgarbage()
     return m
end




function update_model(model, ...)
    list_weight_param, projPoints, wholeSchedule, assignments = ...

    print(assignments)

    numLin = 0 -- Linear modules counter
    numAll = 0 -- All layer counter
    propagate = {} -- whether data needs to be propagated

    -- Flatten model
    flat_model = flatten_model(model)
    -- Update each module separately
    for i = 1, #flat_model.modules do
       local m = flat_model.modules[i]
       local time = sys.clock()
       m = update_module(m)
       collectgarbage()
       flat_model:remove(i)
       flat_model:insert(m, i)
       print('Updated layer ' .. i .. ' module ' .. m.__typename .. ' time ' .. sys.clock() - time)
    end


    return flat_model
end




return update_model



   -- new_features = {}; step = 1
   --
   -- for i = 1, projPoints.data:size(1), step do
   --    new_features[#new_features + 1] = projPoints.data[{{i, i+step-1}}]
   -- end

   -- features = projPoints.data
   -- data = deepcopy(projPoints)




-- update_model:add(m)
-- print(torch.type(m.weight))
--
-- for k = i+1, #net.modules do
--     local n = net.modules[k]
--     if n.__typename == 'nn.Linear' then break end
--     update_model:add(n)
-- end
-- print('update_model ', update_model)
--
-- features:type(torch.type(m.weight))
 -- a = features:type(torch.type(m.weight))
 -- print(torch.type(a))
 -- b = update_model:forward(features)




      --
      --    -- Reset update model
      --    -- update_model = nn.Sequential()
      --    -- update_model:add(cast(nn.Copy(torch.type(features), torch.type(m.weight))))
      --
      --    -- [[choose initialization method]] --
      --    local arg = list_weight_param[numLin].type
      --    local fct = nil
      --    if     arg == 'heuristic'                  then fct = w_init_heuristic
      --    elseif arg == 'xavier'                     then fct = w_init_xavier
      --    elseif arg == 'xaviercaffe'                then fct = w_init_xavier_caffe
      --    elseif arg == 'kaiming'                    then fct = w_init_kaiming
      --    elseif arg == 'simple'                     then fct = w_init_simple
      --    elseif arg == 'kernel' or arg == 'fisher'  then
      --    else
      --       assert(false)
      --    end
      --
      --    -- [[ reset weights ]] --
      --    if arg == 'kernel' then
      --       m = nn.KernelLayer(m.weight:size(2), m.weight:size(1), features, wholeSchedule[numLin], list_weight_param[numLin].scale)
      --    elseif arg == 'fisher' then
      --       m = nn.FisherLayer(m.weight:size(2), m.weight:size(1), data, wholeSchedule[numLin])
      --    else
      --       m:reset(fct(m.weight:size(2), m.weight:size(1)))
      --    end
      --
      --    m.bias:zero()
      --    -- update_model:add(m)
      --    -- print(torch.type(m.weight))
      --    --
      --    -- for k = i+1, #net.modules do
      --    --     local n = net.modules[k]
      --    --     if n.__typename == 'nn.Linear' then break end
      --    --     update_model:add(n)
      --    -- end
      --    -- print('update_model ', update_model)
      --    --
      --    -- features:type(torch.type(m.weight))
      --    -- a = features:type(torch.type(m.weight))
      --    -- print(torch.type(a))
      --    -- b = update_model:forward(features)
      --
      --    print('Linear Layer ' .. numLin .. ' ' .. arg .. ' time ' .. sys.clock() - time)
      -- end
      --
      -- if propagate then
      --    if m.weight ~= nil then
      --       features = features:type(torch.type(m.weight)):clone()
      --    end
      --    features = m:forward(features):clone()
      -- end
      --
      -- data.data = features:clone()
      -- output_net:add(m)

      -- net:remove(i)
      -- net:insert(m, i)
