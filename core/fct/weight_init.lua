-- Different weight initialization methods
-- > model = require('weight-init')(model, 'heuristic')

require 'nn'
require 'image'
require 'utils.aux_fct'
require 'core.layers.KernelLayer'
require 'core.layers.FisherLayer'
require 'core.layers.KLinear'
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


-- Update nn.Sequential with modules with updated weights
function update_sequence(model)

    for i = 1, #model.modules do
       local m = model.modules[i]
       m = update_module(m)
       model:remove(i)
       model:insert(m, i)
    end

    return model
end


function update_module(m)

     if m.__typename == 'nn.Sequential' then
         m = update_sequence(m)
         return m
     end

     numLin = 0

     -- [[ initilization for non-linear layers ]] --
     local arg = list_weight_param[1].type
     local fct = nil
     if     arg == 'heuristic'                  then fct = w_init_heuristic
     elseif arg == 'xavier'                     then fct = w_init_xavier
     elseif arg == 'xaviercaffe'                then fct = w_init_xavier_caffe
     elseif arg == 'kaiming'                    then fct = w_init_kaiming
     elseif arg == 'simple'                     then fct = w_init_simple
     elseif arg == 'kernel'                     then fct = w_init_kaiming
     elseif arg == 'fisher'                     then fct = w_init_kaiming
     else
        assert(false)
     end

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
        numLin = numLin + 1
        local time = sys.clock()

        -- [[choose initialization method]] --
        local arg = list_weight_param[numLin].type
        local fct = nil
        if     arg == 'heuristic'                  then fct = w_init_heuristic
        elseif arg == 'xavier'                     then fct = w_init_xavier
        elseif arg == 'xaviercaffe'                then fct = w_init_xavier_caffe
        elseif arg == 'kaiming'                    then fct = w_init_kaiming
        elseif arg == 'simple'                     then fct = w_init_simple
        elseif arg == 'kernel' or arg == 'fisher'  then fct = nil
        else
           assert(false)
        end

        -- features = torch.cat(new_features, 1)
        -- data.data = features

        -- [[ reset weights ]] --
        if arg == 'kernel' then
           m = nn.KernelLayer(m.weight:size(2), m.weight:size(1), torch.cat(new_features, 1), wholeSchedule[numLin], list_weight_param[numLin].scale)
        elseif arg == 'fisher' then
           data.data = torch.cat(new_features, 1)
           m = nn.FisherLayer(m.weight:size(2), m.weight:size(1), data, wholeSchedule[numLin])
        elseif arg == 'KLinear' then
           m = nn.KLinear(m.weight:size(2), m.weight:size(1), assigments[numLin - 1], opt.KLinearParam)
        else
           m:reset(fct(m.weight:size(2), m.weight:size(1)))
        end


        m.bias:zero()
        print('Linear Layer ' .. numLin .. ' ' .. arg .. ' time ' .. sys.clock() - time)
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

     end

     if propagate and m.__typename ~= 'nn.BatchFlip' and m.weight ~= nil then
        for i = 1, #new_features do
          new_features[i] = new_features[i]:typeAs(m.weight)
        end
        -- features = features:type(torch.type(m.weight)):clone()

        -- [[ prevent memory overflow for complex models]]
        for i = 1, #new_features do
           local curr = m:forward(new_features[i]):clone()
           if curr:nDimension() == 1 then curr = curr:reshape(1, curr:size(1)) end
           if i == 1 then print(m.__typename, curr:size()) end
           new_features[i] = curr
        end

        -- new_features = deepcopy(newnew_features)
        -- newnew_features = nil
        -- collectgarbage()
     end

     return m
end



local function w_init(net, ...)
   list_weight_param, projPoints, wholeSchedule, assignments = ...
   new_features = {}; step = 1

   for i = 1, projPoints.data:size(1), step do
      new_features[#new_features + 1] = projPoints.data[{{i, i+step-1}}]
   end

   -- features = projPoints.data
   -- data = deepcopy(projPoints)
   numLin = 0 -- number of fully-connected layers
   -- local output_net = nn.Sequential()
   -- local update_model = nn.Sequential()

   -- flat_net = flatten_model(net)

   -- Assert if projPoints need to be propagated, i.e. if any of the layers is Kernel or FisherLayer
   for _, val in ipairs(list_weight_param) do
      if val.type == 'kernel' or val.type == 'fisher' then propagate = true  else propagate = false end
   end

   -- -- Flatten data
   -- if opt.model == 'mlp' and propagate then
   --   local dim = 1
   --   for aa = 2, #features:size() do
   --      dim = dim * features:size(aa)
   --   end
   --   features = features:resize(features:size(1), dim)
   -- end


   -- Update all modules
   for i = 1, #net.modules do
      local m = net.modules[i]

      if m.__typename == 'nn.Sequential' then
         m = update_sequence(m)
      else
        net:remove(i)
        net:insert(update_module(m), i)
      end


      -- else
      --
      --
      --
      -- local fct = nil
      -- if     arg == 'heuristic'                  then fct = w_init_heuristic
      -- elseif arg == 'xavier'                     then fct = w_init_xavier
      -- elseif arg == 'xaviercaffe'                then fct = w_init_xavier_caffe
      -- elseif arg == 'kaiming'                    then fct = w_init_kaiming
      -- elseif arg == 'simple'                     then fct = w_init_simple
      -- elseif arg == 'kernel'                     then fct = w_init_kaiming
      -- elseif arg == 'fisher'                     then fct = w_init_kaiming
      -- else
      --    assert(false)
      -- end
      --
      -- if m.__typename == 'nn.SpatialConvolution' then
      --    m:reset(fct(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
      -- elseif m.__typename == 'nn.SpatialConvolutionMM' then
      --    m:reset(fct(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
      -- elseif m.__typename == 'nn.LateralConvolution' then
      --    m:reset(fct(m.nInputPlane*1*1, m.nOutputPlane*1*1))
      -- elseif m.__typename == 'nn.VerticalConvolution' then
      --    m:reset(fct(1*m.kH*m.kW, 1*m.kH*m.kW))
      -- elseif m.__typename == 'nn.HorizontalConvolution' then
      --    m:reset(fct(1*m.kH*m.kW, 1*m.kH*m.kW))
      -- elseif m.__typename == 'nn.TemporalConvolution' then
      --    m:reset(fct(m.weight:size(2), m.weight:size(1)))
      -- elseif m.__typename == 'nn.Linear' then
      --    numLin = numLin + 1
      --    local time = sys.clock()
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
   end
   return net
end


return w_init
