require 'nn'; require 'optim'; require 'paths'; require 'xlua'; require 'math'; image = require 'image'
require 'core.fct.weight_init'; require 'core.fct.test_schedule';
require 'utils.aux_fct'; require 'utils.checks_n_balances';
require 'utils.data_processing'; require 'core.fct.gen_schedule';
require 'utils.monitor'; require 'utils.loggers'
require 'utils.whitening'; c = require 'trepl.colorize'
require 'utils.utils'; require 'utils.allignment'




-- dofile '/homes/mitrovic/Desktop/kernel_init/cifar10/provider.lua'

------------------------------------
---- LOAD DATA AND SET UP EXPERIMENT
------------------------------------
DATASET = 'mnist'

opt = set_experiment_options()
train_data, test_data, classes, projmat = load_data(opt)
projPoints = load_projection_data(train_data)

make_dir_name()
list_weight_param = generate_weight_info(opt)
list_hidden_units = deepcopy(opt.nHiddenUnits)
list_hidden_units[#list_hidden_units + 1] = #classes

print(c.blue '==>' .. ' generating schedule')
schedule, assignments = generate_full_schedule(projPoints, list_hidden_units, list_weight_param)
test_schedule(schedule, assignments, opt, projPoints)



--------------------------
-- LOAD AND UPDATE MODEL
---------------------------
print(c.blue '==>' .. ' updating model')
trainModel = require('models.' .. opt.model)
trainModel = cast(require('core.fct.weight_init_new')(trainModel, list_weight_param, projPoints, schedule, assignments):clone())

if opt.dataset == 'cifar10' then
      trainModel:get(2).updateGradInput = function(input) return end
    if opt.backend == 'cudnn' then
       require 'cudnn'
       cudnn.convert(trainModel:get(3), cudnn)
    end
end

print(c.blue '==> training model', trainModel)
check_model_outputs(trainModel, train_data)

----------------------------------------
--- TRAINING
----------------------------------------
loggers = {}; collectgarbage()
name = make_file_name(list_weight_param)

print(c.blue '==>' .. ' saved at ' .. opt.save)
test_logger = classification_logger(list_hidden_units)

criterion = cast(nn.CrossEntropyCriterion())
confusion = optim.ConfusionMatrix(classes)

train_acc, test_acc = calc_confusion(confusion, trainModel, train_data, test_data)
print(train_acc, test_acc)

-- test_logger:add{train_acc, test_acc}

-- weight_stat_loggers = weight_stat_logger(list_hidden_units)
-- bias_stat_loggers = bias_stat_logger(list_hidden_units)
-- activations_stat_loggers = activations_stat_logger(list_hidden_units)
-- allignment_loggers = allignment_logger(list_hidden_units)
-- allignment_data = make_allignment_data(train_data)


-- Flatten out parameters of network
parameters, gradParameters = trainModel:getParameters()

function train_fct(data)
    trainModel:training()
    param_pre = get_weights_bias(trainModel)
    epoch = epoch or 1

    print(c.blue '==>'.." epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

    local targets = cast(torch.FloatTensor(opt.batchSize))
    local indices = torch.randperm(data.data:size(1)):long():split(opt.batchSize)
    indices[#indices] = nil -- remove last element so that all the batches have equal size

    local tic = torch.tic()
    for t,v  in ipairs(indices) do
       -- xlua.progress(t, #indices)

       local inputs = data.data:index(1, v)
       targets:copy(data.labels:index(1,v))

       local feval = function(x)  -- closure to evaluate f(x) and df/dx
           collectgarbage()
           if x ~= parameters then parameters:copy(x)  end  -- reset parameters
           gradParameters:zero()  -- reset gradients

          local outputs = trainModel:forward(inputs)
          local f = criterion:forward(outputs, targets)   -- evaluate at mini-batch
          local df_dx = criterion:backward(outputs, targets)  -- estimate gradient df/dx
          trainModel:backward(inputs, df_dx)

          -- penalites
          opt.coefL1 = opt.coefL1 or 0
          opt.coefL2 = opt.coefL2 or 0
          if opt.gradNoise then
             local noiseStd = torch.sqrt(opt.gradNoiseNu / ((1 + epoch) ^ opt.gradNoiseGamma))
             noise = torch.Tensor(parameters:size(1)):zero()
             noise:apply(function () return torch.normal(0, noiseStd) end)
          end

          noise = noise or 0
          -- opt.gradNoise = (opt.gradNoise and 1) or 0
          f = f + opt.coefL1 * torch.norm(parameters, 1)
          f = f + opt.coefL2 / 2 * torch.norm(parameters, 2)^2
          gradParameters:add(torch.sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) + noise)

          confusion:batchAdd(outputs, targets)  -- update confusion matrix

          return f, gradParameters   -- return loss and derivative of loss
       end

      --- optimization
      if opt.optimization == 'lbfgs' then
          -- optimization parameters
          optimState = optimState or {
            maxIter = opt.maxIter,
            lineSearch = optim.lswolfe
          }

          -- drop learning rate every "epoch_step" epochs
          if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end

          -- optimization step
          optim.lbfgs(feval, parameters, optimState)

          -- progress
          print('LBFGS step')
          print(' - progress in batch: ' .. t .. '/' .. data.data:size())
          print(' - nb of iterations: ' .. lbfgsState.nIter)
          print(' - nb of function evalutions: ' .. lbfgsState.funcEval)
      --------------------------------------------------------------------
      elseif opt.optimization == 'sgd' then
         -- optimization parameters
         optimState = optimState or {
           learningRate = opt.learningRate,
           weightDecay = opt.weightDecay,
           learningRateDecay = opt.lrDecay,
           momentum = opt.momentum
         }

         -- drop learning rate every "epoch_step" epochs
         if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end


         -- optimization step
         optim.sgd(feval, parameters, optimState)

         -- progress
         -- xlua.progress(t, data.data:size(1))
      --------------------------------------------------------------------
      elseif opt.optimization == 'adagrad' then
         -- optimization parameters
         optimState = optimState or {
           learningRate = opt.learningRate,
           weightDecay = opt.weightDecay
         }

         -- drop learning rate every "epoch_step" epochs
         if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end


         -- optimization step
         optim.adagrad(feval, parameters, optimState)
      --------------------------------------------------------------------
      elseif opt.optimization == 'adam' then
        -- optimization parameters
        optimState = optimState or {
          learningRate = opt.learningRate,
          beta1 = opt.beta1,
          beta2 = opt.beta2,
          epsilon = opt.epsilon
        }

        -- drop learning rate every "epoch_step" epochs
        if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end

        -- optimization step
        optim.adam(feval, parameters, optimState)
      --------------------------------------------------------------------
      else
        error('Unknown optimization method.')
      end

    end -- end of epoch

    -- Classification accuracy
    trainModel:evaluate()
    confusion:updateValids()
    print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
       confusion.totalValid * 100, torch.toc(tic)))
    train_acc = confusion.totalValid * 100
    -- print(confusion)
    confusion:zero()


   -- -- Monitor training
   -- param_post = get_weights_bias(trainModel)   -- parameters of linear layers
   -- rel_diff = rel_change(param_pre, param_post)   -- ratio of weight updates to weights
   -- diff = weight_change(param_pre, param_post)   -- weight updates
   --
   -- for aa = 1, #rel_diff.weights do
   --     local r, p, q = rel_diff.weights[aa], diff.weights[aa], param_pre.weights[aa]
   --     weight_stat_loggers[aa]:add{torch.mean(q), torch.std(q), torch.mean(p), torch.std(p), r}
   -- end
   --
   -- for aa = 1, #rel_diff.bias do
   --     local r, p, q = rel_diff.bias[aa], diff.bias[aa], param_pre.bias[aa]
   --     bias_stat_loggers[aa]:add{torch.mean(q), torch.std(q), torch.mean(p), torch.std(p), r}
   -- end
   --
   -- activations = activations_stat(trainModel)
   -- for aa = 1, #activations do
   --     local r = activations[aa]
   --     activations_stat_loggers[aa]:add{torch.mean(r), torch.std(r), torch.min(r), torch.max(r)}
   -- end
   --
   -- model_allignment(trainModel, allignment_data, allignment_loggers)



   -- calculate score vector for xor data
   -- save/log current net
   -- local filename = paths.concat(, 'mnist.net')
   -- os.execute('mkdir -p ' .. sys.dirname(filename))
   -- if paths.filep(filename) then
   --    os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   -- end
   -- print('<trainer> saving network to '..filename)
   -- torch.save(filename, model)

   epoch = epoch + 1
end


function test_fct(data)
    trainModel:evaluate() -- disable flips, dropouts and batch normalization

    local targets = cast(torch.FloatTensor(opt.batchSize))
    local indices = torch.randperm(data.data:size(1)):long():split(opt.batchSize)
    indices[#indices] = nil -- remove last element so that all the batches have equal size

    local tic = torch.tic()
    for t,v  in ipairs(indices) do
       -- xlua.progress(t, #indices)

       local inputs = data.data:index(1, v)
       local outputs = trainModel:forward(inputs)
       targets:copy(data.labels:index(1,v))
       confusion:batchAdd(outputs, targets)
    end

    confusion:updateValids(); --print(confusion)
    print(('Test accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
       confusion.totalValid * 100, torch.toc(tic)))

    test_logger:add{train_acc, confusion.totalValid * 100}

    -- for j = 1, #trainModel.modules[1] do
    --    local m = trainModel.modules[1].modules[j]
    --    if m.__typename == 'nn.Linear' or m.__typename == 'nn.KernelLayer' then
    --       weights = m.weight
    --       break
    --    end
    -- end

    -- if opt.dataset == 'xor' then
    --     scores = proj_score(weights, projmat)
    --     test_logger:add{train_acc, confusion.totalValid * 100, torch.mean(scores), torch.std(scores)}
    -- else
    --     test_logger:add{train_acc, confusion.totalValid * 100}
    -- end
    --     confusion:zero()
    -- test_logger:style{'-','-'}
    -- test_logger:plot()

    -- local base64im
    -- do
    --   os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
    --   os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
    --   local f = io.open(opt.save..'/test.base64')
    --   if f then base64im = f:read'*all' end
    -- end
    --
    -- local file = io.open(opt.save..'/report.html','w')
    -- file:write(([[
    -- <!DOCTYPE html>
    -- <html>
    -- <body>
    -- <title>%s - %s</title>
    -- <img src="data:image/png;base64,%s">
    -- <h4>optimState:</h4>
    -- <table>
    -- ]]):format(opt.save,epoch,base64im))
    -- for k,v in pairs(optimState) do
    --   if torch.type(v) == 'number' then
    --     file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
    --   end
    -- end
    -- file:write'</table><pre>\n'
    -- file:write(tostring(confusion)..'\n')
    -- file:write(tostring(trainModel)..'\n')
    -- file:write'</pre></body></html>'
    -- file:close()
    --
    -- -- save model every 50 epochs
    -- if epoch % 50 == 0 then
    --   local filename = paths.concat(opt.save, 'model.net')
    --   print('==> saving model to '..filename)
    --   torch.save(filename, trainModel:get(3):clearState())
    -- end

end


local train_time = sys.clock()
epoch = 1
while epoch <= opt.maxEpoch do
   train_fct(train_data)
   test_fct(test_data)

   -- plot errors
--    trainLogger:style{['% mean class accuracy (train set)'] = '-'}
--    test_logger:style{['% mean class accuracy (test set)'] = '-'}
--    trainLogger:plot()
--    test_logger:plot()
end


-- data = {}
-- data.data = torch.Tensor{{1,0,0}, {1.5,0,0}, {0,1,0}, {0,1.5,0}, {0,0,2}}
-- data.labels = torch.Tensor{1,1,2,2,3}

-- list_hidden_units = {3,3}
-- list_weight_param = {{numPos=1, numNeg = 0, type = 'kernel', scale = 100}, {numPos=1, numNeg = 0, type = 'kernel', scale = 100}}

-- schedule, assignments = generate_full_schedule(data, list_hidden_units, list_weight_param)


-- trainModel = nn.Sequential()
-- trainModel:add(nn.Linear(3,3))
-- trainModel:add(nn.ReLU())
-- trainModel:add(nn.Linear(3,3))
-- trainModel:add(nn.LogSoftMax())
--
-- trainModel = cast(require('core.fct.weight_init')(trainModel, list_weight_param, data, schedule, true):clone())
-- print(trainModel)
--
-- name = make_file_name(list_weight_param)
-- allignment_data = make_allignment_data(data)
-- allignment_loggers = allignment_logger(list_hidden_units)
-- model_allignment(trainModel, allignment_data, allignment_loggers)

-- mm = nn.Sequential()
-- mm:add(trainModel:listModules()[2])
-- mm:add(trainModel:listModules()[3])
--
--
-- for i = 1, #allignment_data do
--   allignment_data[i] = mm:forward(cast(allignment_data[i])):clone()
--   print(allignment_data)
-- end


-- model = nn.Sequential()
-- model:add(trainModel:listModules()[4])
-- model:add(trainModel:listModules()[5])
-- bb = model:listModules()[2].weight
--  for key, val in ipairs(allignment_data) do
--      print('class', key, proj_score(bb, val))
--  end
--
-- print('comparison')
-- model_allignment(model, allignment_data, allignment_loggers)
