function vector_unique(input_table)
    local unique_elements = {} --tracking down all unique elements
				local num_elements = {}
    local output_table = {} --result table/vector

    for _, value in ipairs(input_table) do
        unique_elements[value] = true
								num_elements[value] = 0
    end

				for _, value in ipairs(input_table) do
					  num_elements[value] = num_elements[value] + 1
				end

    for key, _ in pairs(unique_elements) do
        table.insert(output_table, key)
    end

    return output_table, num_elements
end


function expand_to_tensor(table, num)

			res = torch.Tensor(num):zero()
			for key, val in pairs(table) do
				   res[key] = res[key] + table[key]
			end

			return res
end





function set_experiment_options()
     -- Loading experiment options
     if DATASET == 'mnist' then
        opt = require('opt.mnist_opt')
        opt.dataset = 'mnist'
     elseif DATASET == 'cifar10' then
        opt = require('opt.cifar10_opt');
        opt.dataset = 'cifar10'
     elseif DATASET == 'xor' then
        opt = require('opt.xor_opt')
        opt.dataset = 'xor'
     elseif DATASET == 'test_data' then
        opt = require('opt.test_data_opt')
        opt.dataset = 'test_data'
     else
        assert(false)
     end
     print('This runs', opt)
     return opt
end


function generate_weight_info(opt)
    wType = {}
    for i = 1, #opt.weightInit do
       local w = {}

       if opt.weightInit[i] == 'kernel' then
            w = {type = opt.weightInit[i], numPos = opt.numPos[i], numNeg = opt.numNeg[i], scale = opt.weightScale}
       elseif opt.weightInit[i] == 'fisher' then
            w = {type = opt.weightInit[i], numPoints = opt.numPoints[i]}
       else
            w = {type = opt.weightInit[i]}
       end

       wType[i] = deepcopy(w)
    end
    return wType
end

function dirtree(dir)
  assert(dir and dir ~= "", "directory parameter is missing or empty")
  if string.sub(dir, -1) == "/" then
    dir=string.sub(dir, 1, -2)
  end

  local function yieldtree(dir)
    for entry in lfs.dir(dir) do
      if entry ~= "." and entry ~= ".." then
          entry=dir.."/"..entry
      	   local attr=lfs.attributes(entry)
      	   coroutine.yield(entry,attr)
      	   if attr.mode == "directory" then
      	     yieldtree(entry)
      	   end
      end
    end
  end

  return coroutine.wrap(function() yieldtree(dir) end)
end

function make_file_name(list_weight_param)
    name = ''; add = ''

    if opt.dropout then add = add .. '-drop' end
    if opt.batchNorm then add = add .. '-bn' end
    if opt.distribution == 'uniform' then cc = '-U' else cc = '-N' end

    for k = 1, #list_weight_param do
       if k > 1 then name = name .. '_' end
       if list_weight_param[k].type == 'kernel' then
          s = 'kernel-p' .. list_weight_param[k].numPos .. '-n' .. list_weight_param[k].numNeg .. '-scl' .. list_weight_param[k].scale .. add .. cc
       elseif list_weight_param[k].type == 'KLinear' then
          s = 'KLinear-' .. opt.KLinearParam .. add
       else
          s = list_weight_param[k].type .. add
       end
       name = name .. s
    end
    return name
end

function make_dir_name()

    if opt.model == 'mlp' then
        architecture = train_data.data:size(2)

        for k = 1, #opt.nHiddenUnits do
           architecture = architecture .. '-' .. opt.nHiddenUnits[k]
        end

        for k = 2, #opt.classHiddenUnits do
           architecture = architecture .. '-' .. opt.classHiddenUnits[k]
        end

        architecture = architecture .. '-' .. #classes .. '/'
    else
        architecture = opt.model .. '/'
    end

    opt_settings = opt.optimization .. '/' .. '*lr*' .. opt.learningRate .. '*mom*' .. opt.momentum .. '*lrd*' .. opt.lrDecay .. '*L1*' .. opt.coefL1 .. '*L2*' .. opt.coefL2


    if opt.dataset == 'xor' then
       opt.save = opt.save .. os.date("%Y_%m_%d") .. '/' .. opt.dataset  .. architecture
       .. 'overlap*' .. opt.overlap .. 'dim*' .. opt.nDim .. 'noise*' .. opt.noiseFactor ..
               'dist*' .. opt.distribution .. '/' .. opt_settings
    else
        opt.save = opt.save .. os.date("%Y_%m_%d") .. '/' .. opt.dataset .. '/' .. architecture .. opt_settings
    end

    opt.save =  opt.save .. '/' .. opt.end_path
end
