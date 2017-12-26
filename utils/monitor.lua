require 'nn'
require 'math'
require 'gnuplot'

function find_transfer_layers(model)
	    model = flatten_model(model)
					ind = {}
					for key, val in ipairs(model:listModules()) do
						   if val.__typename == 'nn.ReLU' then
													ind[#ind + 1] = key
									end
					end

				return ind
end


function flatten_model(model)
					-- Flatten model
					flat_model = nn.Sequential()
					for key, val in pairs(model:listModules()) do
									if val.__typename ~= 'nn.Sequential' then
													flat_model:add(val)
									end
					end

					return flat_model
end


function get_weights_bias(model)
	    param, W, b, t = {}, {}, {}, {}
	    linear_modules = model:findModules('nn.Linear')
					kernel_modules = model:findModules('nn.KernelLayer')
					fisher_modules = model:findModules('nn.FisherLayer')

					for k = 1, #linear_modules do
						  local m = linear_modules[k]
						  W[#W + 1] = torch.Tensor(m.weight:size()):copy(m.weight)
								b[#b + 1] = torch.Tensor(m.bias:size()):copy(m.bias)
								t[#t+1] = m.__typename
					end

					for k = 1, #kernel_modules do
								local m = kernel_modules[k]
								W[#W + 1] = torch.Tensor(m.weight:size()):copy(m.weight)
								b[#b + 1] = torch.Tensor(m.bias:size()):copy(m.bias)
								t[#t +1] = m.__typename
					end

					for k = 1, #fisher_modules do
						   local m = fisher_modules[k]
									W[#W + 1] = torch.Tensor(m.weight:size()):copy(m.weight)
									b[#b + 1] = torch.Tensor(m.bias:size()):copy(m.bias)
									t[#t +1] = m.__typename
					end


					param.weights, param.bias, param.type = W, b, t
     return param
end

function weight_histogram(model)
	    param = get_weights_bias(model)
					weights = param.weights

					for key, val in ipairs(weights) do
					    print('Layer ' .. key)
					    print(torch.min(val), torch.mean(val), torch.max(val))
					end

					for layer, val in ipairs(weights) do
						  gnuplot.figure()
								gnuplot.title('Layer ' .. layer)
						  gnuplot.hist(val)
					end
end


function activations_stat(model)

					num_layers = #model:findModules('nn.ReLU') + #model:findModules('nn.LogSoftMax')

					data = test_data.data[{{1, 1000}, {}}]
					activations = {}

					-- Flatten model
					flat_model = flatten_model(model)

					-- Activations at different levels averaged over 300 test points
					for a = 1, #flat_model.modules do
								local m = flat_model.modules[a]
								data = m:forward(data)
								if m.__typename == 'nn.ReLU' or m.__typename == 'nn.LogSoftMax' then
									  activations[#activations + 1] = data
								end
					end

					return  activations
end


function weight_change(param_pre, param_post)
						diff = {}
						for key, val in pairs(param_pre) do
										diff[key] = {}
										for i = 1, #val do
														a, b = param_pre[key][i], param_post[key][i]
														diff[key][i] = a - b
										end
						end

						return diff
end


function rel_change(param_pre, param_post)
	     rel_diff = {}
						diff = weight_change(param_pre, param_post)
						for key, val in pairs(param_pre) do
							   rel_diff[key] = {}
										for i = 1, #val do
											   a = param_pre[key][i]
														d = diff[key][i]:view(diff[key][i]:nElement())
											   rel_diff[key][i] = torch.norm(d) / torch.norm(a:view(a:nElement()))
														                   -- torch.cdiv(a - b, b)
										end
						end
						return rel_diff
end

function model_probs(model, data)
	   local prop_data = model:forward(data.data)

				return torch.exp(prop_data)
end


function TableConcat(t1,t2)
    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
end

function split_data_into_classes(data)

				local features = data.data
				local labels = data.labels
				local data_split_classes = {}
				local ind_split_classes = {}

				for i = 1, labels:max() do
					   n, m = labels[labels:eq(i)]:size(1), features:size(2)
					   data_split_classes[i] = torch.Tensor(n,m):zero()
								ind_split_classes[i] = {}
				end

				for a = 1, labels:max() do

							ind = torch.range(1, labels:nElement())[labels:eq(a)]
							data_split_classes[a] = features:index(1, ind:long())
							ind_split_classes[a] = ind

							-- j = 1
							-- ind = torch.totable(labels[labels:eq(a)])
							-- for _, val in ipairs(ind) do
							-- 	  data_split_classes[a][j]	= features:index(1, torch.LongTensor{val})
							-- 			j = j + 1
							--    ind_split_classes[a] = TableConcat(ind_split_classes[a], {val})
							-- 	end
				end
				return data_split_classes, ind_split_classes
end


-- Calculate model output probabilities according to class
function model_probs_class(model, data)

					local features = data.data
					local labels = data.labels

					local data_split_classes = {}
					local probs_per_class = {}

					data_split_classes, _ = split_data_into_classes(data)

					for i = 1, labels:max() do
						  -- local d = nn.JoinTable(1):forward(data_split_classes[i])
								d = data_split_classes[i]:clone()
								local log_probs = model:forward(cast(d))
								probs_per_class[i] = torch.exp(log_probs)
					end

					return probs_per_class
end

function model_layer_output_class(model, data)

					local features = data.data
					local labels = data.labels

					local data_split_classes = {}
					outputs = {}
					data_split_classes, _ = split_data_into_classes(data)

					-- Flatten model
					flat_model = flatten_model(model)

					-- Activations at different layers averaged over 1000 training points
					for a, _ in ipairs(flat_model.modules) do
								outputs[a] = {}
					end

					for b = 1, #data_split_classes do
						   local ind = math.floor(data_split_classes[b]:size()[1] / 5)
									local dat = data_split_classes[b][{{1, ind}, {}}]

									for a, m in ipairs(flat_model.modules) do
													dat = m:forward(dat):clone()
									    outputs[a][b] = dat
									end
					end

					return outputs
end







function calc_confusion(confusion, model, train_data, test_data)

			model:evaluate()

			-- Train set
			for a = 1, train_data.data:size(1), 1000 do
						confusion:batchAdd(model:forward(train_data.data[{{a, math.min(a+1000, train_data.data:size(1))}}]),
																				train_data.labels[{{a, math.min(a+1000, train_data.data:size(1))}}])
			end

			confusion:updateValids()
			train_acc = confusion.totalValid * 100
	  print(confusion)
			confusion:zero()

			-- Test set
			for a = 1, test_data.data:size(1), 1000 do
			   confusion:batchAdd(model:forward(test_data.data[{{a, math.min(a + 1000, test_data.data:size(1))}}]),
			            test_data.labels[{{a, math.min(a + 1000, test_data.data:size(1))}}])
			end

			confusion:updateValids()
			test_acc = confusion.totalValid * 100
			print(confusion)
			confusion:zero()

			return train_acc, test_acc
end
