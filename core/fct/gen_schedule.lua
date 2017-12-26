require 'nn'
require 'utils.aux_fct'
require 'utils.monitor'

function labels(projPoints)
    -- Encode labels in tensor num_labels * num_points
    -- i,j=1 j-th point belongs to class i

    numProjPoints = projPoints.data:size(1)
    d_outputs = projPoints.labels:max()

    p = torch.Tensor(d_outputs, numProjPoints):zero()
    for i = 1, projPoints.labels:size(1) do
        p[{{projPoints.labels[i]}, {i}}] = 1
    end

    return p
end


function convert_assignments(assignments)
   local output = {}

   for j, val in ipairs(assignments) do
         if output[val] == nil then output[val] = {} end
         output[val][#output[val] + 1] = j
   end

   return output
end


function generate_full_schedule(projPoints, list_hidden_units, list_weight_param)
      -- Generate the schedule (indices and signs) for all layers of the network

      schedule = {}; assignments = {}
      local p = labels(projPoints) -- encoding of labels vs. points

      for i = 1, #list_hidden_units  do

         local hidden = list_hidden_units[i]
         local weight_param = list_weight_param[i]

         -- Generate schedule for layerwise
         local time = sys.clock()

         if list_weight_param[i].type == 'kernel' then
           schedule[#schedule + 1], assignments[#assignments + 1]
                 = kernel_schedule(projPoints, hidden, weight_param, p)
         elseif list_weight_param[i].type == 'fisher' then
            schedule[#schedule + 1] = fisher_schedule(projPoints, hidden, weight_param)
         else
            schedule[#schedule + 1], assignments[#assignments + 1] = {}, {}
         end

         print('Layer', i, weight_param.type, 'time', sys.clock() - time)
     end

     return schedule, assignments
end



function fisher_schedule(projPoints, nNeurons, weight_param)
     local numPointsPerGroup = weight_param.numPoints
     local numClasses = projPoints.labels:max()
     local numGroups = math.ceil(nNeurons / numClasses)

     -- make schedule for layer
     local sc = {indices = {}, signs = {}}

     -- Just random points no balancing classes
     -- for i = 1, numGroups do
     --      a = torch.Tensor(numPointsPerGroup):random(1, opt.trainSize)
     --      sc.indices[i] = deepcopy(a)
     -- end

     -- Balancing classes
     _, ind_split_classes = split_data_into_classes(projPoints)

     for i = 1, numGroups do
         a = {}
         for j = 1, numClasses do
            local ind = torch.Tensor(numPointsPerGroup / numClasses):random(1, ind_split_classes[j]:size(1))
            a[j] = torch.Tensor(ind_split_classes[j]):index(1, ind:long())
         end
         sc.indices[i] = deepcopy(nn.JoinTable(1):forward(a))
     end

     return sc
end


function kernel_schedule(projPoints, nNeurons, weight_param, p)
    -- number of pos/neg points per neuron per layer
    local numPointsPerNeuron =
               weight_param.numNeg * (projPoints.labels:max() - 1) + weight_param.numPos
    -- number of neurons associated with one class
    local neuronsPerClass = nNeurons / projPoints.labels:max()

    local sc = {indices={},signs={}} -- make schedule for every layer
    local assignments = {}

    -- number of draws per class
    local numDrawsClass = {}
    for b = 1, projPoints.labels:max() do
        numDrawsClass[b] = {}
        for j = 1, projPoints.labels:max() do
            if b == j then
               numDrawsClass[b][j] = weight_param.numPos
            else
               numDrawsClass[b][j] = weight_param.numNeg
            end
        end
    end


    for j = 1, nNeurons do

       local ind = torch.Tensor(numPointsPerNeuron):zero()
       local sign = torch.Tensor(numPointsPerNeuron):zero()

       -- for which class is the neuron a feature detector
       local neuronCurrentClass = math.min(math.floor((j-1) / neuronsPerClass) + 1, projPoints.labels:max())
       assignments[#assignments + 1] = neuronCurrentClass

       -- draw indices of schedule and set signs
       local dummyIndex = 1
       for k = 1, projPoints.labels:max() do
           local numPoints = numDrawsClass[k][neuronCurrentClass]
           -- schedule indices
           if numPoints > 0 then
               ind[{{dummyIndex, dummyIndex + numPoints -1}}] = torch.multinomial(p[k], numPoints)

               -- schedule signs
               if k == neuronCurrentClass then
                  sign[{{dummyIndex, dummyIndex + numPoints -1}}] = torch.Tensor(numPoints):fill(1)
               else
                  sign[{{dummyIndex, dummyIndex + numPoints -1}}] = torch.Tensor(numPoints):fill(-1)
               end
           end

           -- update index
           dummyIndex = dummyIndex + numPoints
       end

        sc.indices[j] = ind
        sc.signs[j] = sign

        if opt.shuffle then
           math.randomseed(1)
           shuffle(sc.indices)
           math.randomseed(1)
           shuffle(sc.signs)
           math.randomseed(1)
           shuffle(assignments)
        end
    end

    return sc, convert_assignments(assignments)
end




function set_alphas(alpha, schedule)
   -------
   -- This function zeros out entries of alpha corresponding to schedule
   -------
   -- alpha - torch.Tensor(num_neurons, #schedule[1]) with non-zero entries
   -- schedule - table where each entry contains the indices that won't be zeroed out

   assert(torch.type(alpha) == 'torch.DoubleTensor', 'alpha must be a torch.DoubleTensor.')
   assert(type(schedule) == 'table', 'schedule must be a table.')

   local function zero_out(row_alpha, scheduleIndices, scheduleSigns)

       -- which indices should not be zeroed out; expand scheduleSigns
       local indices = torch.Tensor(row_alpha:size(1)):zero()
       local schedSigns = torch.Tensor(row_alpha:size(1)):zero()
       for i = 1, scheduleIndices:size(1) do
          indices[scheduleIndices[i]] = 1
          schedSigns[scheduleIndices[i]] = scheduleSigns[i]
       end

       for j = 1, row_alpha:size(1) do
           if indices[j] ~= 1 then
              row_alpha[j] = 0
           end
       end
       return torch.cmul(row_alpha, schedSigns)
   end

   for i = 1, alpha:size(1) do
      alpha[i] = zero_out(alpha[i], schedule.indices[i], schedule.signs[i])
   end

   return alpha
end
