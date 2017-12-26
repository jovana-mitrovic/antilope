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


init = torch.Tensor(5, 10)
t = {'heuristic', 'xavier', 'xaviercaffe', 'kaiming'}

opt = set_experiment_options()

for k, val in ipairs(t) do
   opt.weightInit = {val, val}
   for p = 1, 10 do

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
    -- test_logger = classification_logger(list_hidden_units)

    criterion = cast(nn.CrossEntropyCriterion())
    confusion = optim.ConfusionMatrix(classes)

    train_acc, test_acc = calc_confusion(confusion, trainModel, train_data, test_data)

    init[{{k}, {p}}] = test_acc

   end
end

print(init)

print(torch.mean(init, 2))
print(torch.std(init,2))
