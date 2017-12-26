require 'nn'

function generate_classifier(classSchedule, projPoints, list_weight_param)

   -- classifier = nn.Sequential()
   -- if opt.dropout then classifier:add(nn.Dropout(0.5)) end

   for i = 2, #opt.classHiddenUnits do
       classifier:add(nn.Linear(opt.classHiddenUnits[i - 1], opt.classHiddenUnits[i]))

       -- if opt.batchNorm then classifier:add(nn.BatchNormalization(opt.classHiddenUnits[i])) end
       -- classifier:add(nn.ReLU())
       -- if opt.dropout then classifier:add(nn.Dropout(0.3)) end
   end

   if list_weight_param[#list_weight_param].type == 'KLinear' then
       require 'core.layers.KLinear'
       classifier:add(nn.KLinear(opt.classHiddenUnits[#opt.classHiddenUnits], #classes, detector_assignments[#detector_assignments], opt.KLinearParam))
   else
       classifier:add(nn.Linear(opt.classHiddenUnits[#opt.classHiddenUnits], #classes))
   end

   -- if opt.batchNorm then classifier:add(nn.BatchNormalization(#classes)) end
   -- classifier:add(nn.LogSoftMax())
   -- classifier = require('core.fct.weight_init')(classifier, list_weight_param, projPoints, classSchedule, true):clone()

   -- return classifier
-- end

-- return generate_classifier
