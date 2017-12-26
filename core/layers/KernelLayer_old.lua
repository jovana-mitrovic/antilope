require 'nn'  -- require 'cutorch'
require 'utils.stats'


local KernelLayer, parent = torch.class('nn.KernelLayer', 'nn.Linear')

   function KernelLayer:__init(inputSize, outputSize, projData, schedule, weightScale)

      assert(torch.isTensor(projData), 'Provide projection points as tensor')
      assert(torch.type(schedule) == 'table', 'Provide schedule as table of indices.')
      assert(outputSize, #schedule.indices, 'Length of schedule must be equal to the output dimension.' )

      parent.__init(self, inputSize, outputSize)

      n_proj, d_proj = projData:size(1), projData:size(2)
      alpha = torch.Tensor(outputSize, n_proj)

      -- calculate the standard deviation of the alpha's
      local projPoints_square = torch.cmul(projData, projData) -- element-wise multiplication
      projPoints_square = torch.cmul(projPoints_square, projPoints_square)
      local sum_mat = cast(torch.Tensor(#schedule.indices, projData:size(2)):zero():float())

      for s = 1, #schedule.indices do
          local schedule_entry, sum = schedule.indices[s], torch.Tensor(projData:size(2)):zero()
          for j = 1, schedule_entry:size(1) do
              sum_mat[s] = sum_mat[s] + cast(projPoints_square[schedule_entry[j]]:float())
          end
      end
      mean = torch.mean(sum_mat)
      num_ksi = #schedule.indices[1]

      if opt.distribution == 'normal' then
         method = torch.randn
         factor = math.sqrt( 1 / (weightScale * d_proj * mean) )  -- weightScale * d_proj * mean before
      elseif opt.distribution == 'uniform' then
         method = torch.rand
         factor = math.sqrt( 12 / (weightScale * d_proj * mean) )
      end

      alpha = method(outputSize, n_proj); alpha:mul(factor)
      alpha = set_alphas(alpha, schedule)  -- zero out entries according to schedules

      -- if self.bias then
      --    self.bias = method(self.bias:nElement()); self.bias:mul(factor)
      -- end


      self.bias:zero()
      self.weight = torch.Tensor(outputSize, projData:size(2)):zero()
      self.weight:addmm(alpha, projData:double())

end

--  ***** DEPRECATED CODE ****** --
-- -- populate alphas
-- for i = 1, alpha:size(1) do
  -- local M_a = schedule.indices[i]:size(1) -- number of active alphas (i.e. not zeroed out)

  -- mean of kernel matrix of projection points
  -- local av = torch.mean(torch.cmul(projPoints, projPoints))

  -- local A = torch.Tensor(d_proj, #schedule.indices)
  -- for s = 1, #schedule.indices do
  --     local schedule_entry = schedule.indices[s]
  --     for j = 1, schedule_entry:size(1) do
  --         for t = 1, d_proj do
  --            A[{{t},{s}}] = A[{{t},{s}}] + projPoints_square[{{schedule_entry[j]},{t}}]
  --         end
  --     end
  -- end
  --
  -- local av = torch.mean(A) / (schedule.indices[1]:size(1) * d_proj)

  -- local projPoints_square = torch.cmul(projPoints, projPoints)
  -- sum = sum + torch.sum(projPoints_square:select(1, schedule_entry[j]))
  -- local M_a = schedule.indices[1]:size(1)
  -- local av = sum / (schedule.indices[1]:size(1) * d_proj)

  -- alpha_stdev = math.sqrt( 4 / (projPoints:size(2) * av * M_a) )
  -- local alpha_stdev = math.sqrt( 12 / (weightScale * d_proj * av * M_a) )
  -- for i = 1, alpha:size(1) do
  --    alpha:select(1, i):apply(function () return
  --                 torch.uniform(0, alpha_stdev) end)
  -- end

  --       local weight_noise = torch.Tensor(outputSize, projPoints:size(2)):zero()
        -- for i = 1, weight_noise:size(1) do
        --    for j = 1, weight_noise:size(2) do
        --       weight_noise[{{i},{j}}] = torch.normal(0,1)
        --    end
        -- end
        -- weight_noise = weight_noise:double()
  -- self.weight = torch.cmul((1 + 0.1 * weight_noise), self.weight)
