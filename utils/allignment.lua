require 'utils.monitor'
require 'utils.loggers'
require 'math'


function normalize_vec(vec)
     return vec / torch.norm(vec)
end

function normalize_rows(matrix)
    for j = 1, matrix:size(1) do
        matrix[j] = normalize_vec(matrix[j])
    end
    return matrix
end

function proj_score(mat, projmat)
    -- mat - row-wise, torch.Tensor
    -- proj_vectors - row-wise, torch.Tensor

    assert(mat:size(2) == projmat:size(2), 'Input dimensions of mat and projection vectors do not agree.')
    torch.setdefaulttensortype('torch.DoubleTensor')
    mat = mat:type('torch.DoubleTensor')

    pm = projmat:clone()
    pm = normalize_rows(pm)

    -- orthonormalize vectors
    for k = 2, pm:size(1) do
       for t = 1, pm:size(1) do
           if k ~= t then
              pm[{k}] = pm[{k}] - torch.dot(pm[{k}], pm[{t}])/ math.max(torch.norm(pm[{t}]), 1e-04) * pm[{t}]
           end
       end
    end

    local res = torch.Tensor(mat:size(1)):zero()
    for j = 1, mat:size(1) do
        w = normalize_vec(mat:select(1, j))
        local result = torch.Tensor(w:size(1)):zero()
        for k = 1, pm:size(1) do
            v = pm:select(1, k):type(w:type())
            if torch.norm(v) ~= 0 then
                v = normalize_vec(v)
                result:add(torch.dot(w, v)/ torch.norm(v) * v)
            end
        end
        res[j] = torch.norm(result) / torch.norm(w)
    end
    return res
end


function make_allignment_data(train_data)
    -- Set data according which to measure allignment

    torch.manualSeed(123)
    data_split_classes, ind_split_classes = split_data_into_classes(train_data)
    rand_ind = torch.Tensor(100):random(train_data.data:size(1)):long()
    random_subsample = train_data.data:index(1, rand_ind)

    allignment_data = {}
    for key, val in ipairs(data_split_classes) do
        ind = torch.Tensor(100):random(val:size(1)):long()
        allignment_data[#allignment_data + 1] = val:index(1, ind)
    end

    allignment_data[#allignment_data + 1] = random_subsample

    return allignment_data
end



function weight_allignment(weights, allignment_data, loggers)

    results = torch.Tensor(weights:size(1), #allignment_data):zero()
    for key, val in ipairs(allignment_data) do
        results[{{},{key}}] = proj_score(weights, val)
    end

    print(results)

    r = torch.totable(results)
    for j = 1, weights:size(1) do
          loggers[j]:add{r[j][1], r[j][2], r[j][3], r[j][4], r[j][5],
           r[j][6], r[j][7], r[j][8], r[j][9], r[j][10], r[j][11]}
    end
end


function model_allignment(model, allignment_data, log)
   dat = deepcopy(allignment_data)
   mod = flatten_model(model):clone()
   transfer_layer_ind = find_transfer_layers(mod)
   param = get_weights_bias(mod)
   weights_list = param.weights

   start_ind, t = 1, 1
   for key, val in ipairs(weights_list) do
       weight_allignment(val, dat, log[key])
       end_ind = transfer_layer_ind[t] or #mod:listModules()
       for a = start_ind, end_ind do
           m = mod:listModules()[a]
           if m.__typename ~= 'nn.Sequential' then
               print(m.__typename)
               for i = 1, #dat do
                  dat[i] = m:forward(cast(dat[i])):clone()
               end
           end
        end

        if end_ind ~= #mod:listModules() then
            start_ind = transfer_layer_ind[t] + 1
            t = t + 1
        end
   end
end
