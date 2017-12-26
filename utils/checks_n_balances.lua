require 'utils.monitor'
require 'math'
c = require 'trepl.colorize'
require 'utils.utils'

---------------------
--- MODEL OUTPUTS ---
---------------------

function check_model_outputs(model, data)

   local labels = data.labels

			-- Output of model according to class
			local outputs = model_layer_output_class(model, data)
   local input_data_class, _ = split_data_into_classes(data)
   local flat_model = flatten_model(model)
   local param = get_weights_bias(model)

   -- -- Weights of modules
   -- print(c.blue '==> printing weights')
   -- for a, val in ipairs(flat_model.modules) do
   --     print('Layer ' .. a, val.__typename)
   --     print(val.weight)
   -- end
   --
   --  -- Input data
   --  print(c.blue '==> inptu data according to class' )
   --  for a, val in ipairs(input_data_class) do
   --     print('Class ' .. a, torch.mean(val, 1))
   --  end
   --
   -- -- Outputs of layers according to class
   -- print(c.blue '==> module outputs according to class')
			-- for a, value in ipairs(outputs) do
   --    	for k, val in ipairs(value) do
   --       print(flat_model.modules[a].__typename)
   --       print('Class ' .. k, torch.mean(val, 1))
			--    end
   -- end



   -- Output probabilities according to classes
			probs_per_class = model_probs_class(model, data)
			for i = 1, labels:max() do
				   print('Class ' .. i, torch.mean(probs_per_class[i], 1))
			end

			-- -- Output probabilities of data
			-- probs = model_probs(trainModel, train_data)

end


----------------
--- SCHEDULE ---
----------------
function test_schedule(schedule, assignments, opt, projData)

	    if schedule_correct(schedule, assignments, opt, projData) then
						  print(c.blue '==> ' .. 'schedule checked')
					else
						  print(c.blue '==> ' .. 'WARNING!!! SCHEDULE NOT CORRECT !!!')
					end

					return
end

function find_element(table, element)
   for key, val in pairs(table) do
      for k, v in pairs(val) do
         if v == element then
            return key
         end
       end
    end

     return key
end





function schedule_correct(schedule, assignments, opt, projData)
    -- assignments - table; each layer one entry; each entry a table with entries corresponding to
    -- classes and each entry being a table of neurons assigned to that class
				local correct = true
    local num_classes = projData.labels:max()

    for key, type in ipairs(opt.weightInit) do
       local sch = schedule[key]

       if type == 'kernel'  then
          local as = assignments[key]
          -- print(layer, as)

  								for neuron, s in ipairs(sch.indices) do
              local neuron_class = find_element(as, neuron) -- which class does the neuron encode
              -- print(layer, neuron, neuron_class)
              local l = projData.labels:index(1, s:long()) -- extract labels
              local pos_counter = opt.numPos[key]
              local neg_counter = opt.numNeg[key] * (num_classes - 1)


              for k = 1, l:size(1) do
                  if l[k] == neuron_class then
                     if  schedule[key].signs[neuron][k] == 1 then
                         pos_counter = pos_counter - 1
                     else
                        correct = false
                     end
                  else
                     if schedule[key].signs[neuron][k] == -1 then
                        neg_counter = neg_counter - 1
                     else
                        correct = false
                     end

                  end

              end

              if pos_counter ~= 0 or neg_counter ~= 0 then
                 correct = false
              end
          end
       elseif type == 'fisher' then
           -- Check all schedule entries
           for _, s in ipairs(sch.indices) do
              local l = projData.labels:index(1, s:long()) -- extract labels
              local counter = (opt.numPoints[key] / num_classes) * torch.ones(num_classes)
              -- Check that there are enough points from each class
              for k = 1, num_classes do
                 if l[l:eq(k)]:size(1) ~= counter[k] then
                      correct = false
                 end
              end
           end
       end
  		end

				return correct
end




-- local _, num_elements = vector_unique(torch.totable(l))
--
-- num_elements = expand_to_tensor(num_elements, num_classes)
--
-- for b = 1, num_classes do
-- 	   if b == neuron_class then
-- 					  correct = (p == num_elements[neuron_class])
-- 			 else
-- 					  correct = (n == num_elements[b])
-- 				end
--
-- 				if not correct then
-- 					 print(layer, neuron, b)
-- 						return false
-- 		  end
-- end


	-- weight_histogram(trainModel)

	-- param = get_weights_bias(trainModel)
	-- for key, val in ipairs(param.weights) do
	--     print(val[{{1}, {}}])
	-- end

	-- if opt.dataset == 'xor' then
	--
	--     for j = 1, #trainModel.modules[1] do
	--        local m = trainModel.modules[1].modules[j]
	--        if m.__typename == 'nn.Linear' or m.__typename == 'nn.KernelLayer' then
	--           weights = m.weight
	--           break
	--        end
	--     end
	--
	--     scores = proj_score(weights, projmat)
	--     test_logger:add{train_acc, test_acc, torch.mean(scores), torch.std(scores)}
	-- else
	--     test_logger:add{train_acc, test_acc}
	-- end
	--
	--
	--
	-- d = train_data.data:clone()
	--
	-- for i = 1, #trainModel.modules do
	--   for j = 1, #trainModel.modules[i] do
	--       local m = trainModel.modules[i].modules[j]
	--       d = m:forward(d)
	--       print(m.__typename, torch.mean(d), torch.std(d))
	--       if m.__typename == 'nn.Linear' or m.__typename == 'nn.KernelLayer' then
	--            print('mean weight', torch.mean(m.weight)); end
	--   end
	-- end
	-- print('probs', torch.mean(torch.exp(d), 1), torch.std(torch.exp(d),1))
