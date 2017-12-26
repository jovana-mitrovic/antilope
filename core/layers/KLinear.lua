local KLinear, parent = torch.class('nn.KLinear', 'nn.Module')

   function KLinear:__init(inputSize, outputSize, assignment, KLinearParam, bias)
       parent.__init(self)
       local bias = ((bias == nil) and true) or bias
       self.weight = torch.Tensor(outputSize, inputSize):fill(KLinearParam)
       self.gradWeight = torch.Tensor(outputSize, inputSize):zero()
       if bias then
          self.bias = torch.Tensor(outputSize):zero()
          self.gradBias = torch.Tensor(outputSize):zero()
       end
       self:setWeights(assignment)
   end

   function KLinear:noBias()
      self.bias = nil
      self.gradBias = nil
      return self
   end

   function KLinear:setWeights(assignment)
       assert(#assignment == self.weight:size(1), 'Provide appropriate number of detector assigments.')
       assert(torch.type(assignment) == 'table', 'Provide detector assigments as table of tables of indices.')

       for k = 1, self.weight:size(1) do
         for _,j in ipairs(assignment[k]) do
             self.weight[{{k}, {j}}] = torch.abs(self.weight[{{k}, {j}}])
         end
       end

       if self.bias then
          for i = 1, self.bias:nElement() do
             self.bias[i] = 0
          end
       end

       return self
   end

   local function updateAddBuffer(self, input)
      local nframe = input:size(1)
      self.addBuffer = self.addBuffer or input.new()
      if self.addBuffer:nElement() ~= nframe then
         self.addBuffer:resize(nframe):fill(1)
      end
   end

   function KLinear:updateOutput(input)
      if input:dim() == 1 then
         self.output:resize(self.weight:size(1))
         if self.bias then self.output:copy(self.bias) else self.output:zero() end
         self.output:addmv(1, self.weight, input)
      elseif input:dim() == 2 then
         local nframe = input:size(1)
         local nElement = self.output:nElement()
         self.output:resize(nframe, self.weight:size(1))
         if self.output:nElement() ~= nElement then
            self.output:zero()
         end
         updateAddBuffer(self, input)
         self.output:addmm(0, self.output, 1, input, self.weight:t())
         if self.bias then self.output:addr(1, self.addBuffer, self.bias) end
      else
         error('input must be vector or matrix')
      end

      return self.output
   end

   function KLinear:updateGradInput(input, gradOutput)
      if self.gradInput then

         local nElement = self.gradInput:nElement()
         self.gradInput:resizeAs(input)
         if self.gradInput:nElement() ~= nElement then
            self.gradInput:zero()
         end
         if input:dim() == 1 then
            self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
         elseif input:dim() == 2 then
            self.gradInput:addmm(0, 1, gradOutput, self.weight)
         end

         return self.gradInput
      end
   end

   function KLinear:accGradParameters(input, gradOutput, scale)
      scale = scale or 1
      if input:dim() == 1 then
         self.gradWeight:addr(scale, gradOutput, input)
         if self.bias then self.gradBias:add(scale, gradOutput) end
      elseif input:dim() == 2 then
         self.gradWeight:addmm(scale, gradOutput:t(), input)
         if self.bias then
            -- update the size of addBuffer if the input is not the same size as the one we had in last updateGradInput
            updateAddBuffer(self, input)
            self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
         end
      end
   end

   -- we do not need to accumulate parameters when sharing
   KLinear.sharedAccUpdateGradParameters = KLinear.accUpdateGradParameters

   function KLinear:clearState()
      if self.addBuffer then self.addBuffer:set() end
      return parent.clearState(self)
   end

   function KLinear:__tostring__()
     return torch.type(self) ..
         string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1)) ..
         (self.bias == nil and ' without bias' or '')
   end
