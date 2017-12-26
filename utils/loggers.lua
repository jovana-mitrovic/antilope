require 'optim'

function classification_logger()
					-- Classification accuracy
					testLogger = optim.Logger(paths.concat(opt.save, name .. '.txt'))
					testLogger:setNames{'mean_class_accuracy_(train set)', 'mean_class_accuracy_(test set)'}
					testLogger.showPlot = false

					return testLogger
end

function weight_stat_logger(list_hidden_units)
					-- Weight statistics
					weight_stat_loggers = {}
					for k = 1, #list_hidden_units do
									weight_stat_loggers[k] = optim.Logger(paths.concat(opt.save .. '/weight ' .. k .. '/', name .. '.txt'))
									weight_stat_loggers[k]:setNames{'mean', 'std', 'mean_change', 'std_change', 'ratio'}
									weight_stat_loggers[k].showPlot = false
					end

					return weight_stat_loggers
end


function bias_stat_logger(list_hidden_units)
					-- Bias statistics
					bias_stat_loggers = {}
					for k = 1, #list_hidden_units do
									bias_stat_loggers[k] = optim.Logger(paths.concat(opt.save .. '/bias ' .. k .. '/', name .. '.txt'))
									bias_stat_loggers[k]:setNames{'mean', 'std', 'mean_change', 'std_change', 'ratio'}
									bias_stat_loggers[k].showPlot = false
					end

					return bias_stat_loggers
end


function activations_stat_logger(list_hidden_units)
					-- Activations statistics
					activations_stat_loggers = {}
					for k = 1, #list_hidden_units do
									activations_stat_loggers[k] = optim.Logger(paths.concat(opt.save .. '/activations ' .. k .. '/', name .. '.txt'))
									activations_stat_loggers[k]:setNames{'activations_mean', 'activations_std', 'activations_min', 'activations_max'}
									activations_stat_loggers[k].showPlot = false
					end

					return activations_stat_loggers
end


function allignment_logger(list_hidden_units)
				allignment_loggers = {}
				for k = 1, #list_hidden_units do
					  allignment_loggers[k] = {}
				end

				for ind, val in ipairs(list_hidden_units) do
					  for i = 1, val do
								  n = 'allignment/layer_' .. ind .. '_row_' .. i .. '/' .. name .. '.txt'
					     allignment_loggers[ind][i] = optim.Logger(paths.concat(opt.save, n))
										allignment_loggers[ind][i]:setNames{'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8', 'class 9', 'class 10', 'random subsample'}
							end
				end

				return allignment_loggers
end								
