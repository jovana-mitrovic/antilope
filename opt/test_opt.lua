-- Test command line inputs

function test_cmd(opt)

   assert(#opt.nHiddenUnits > 0, 'Number of hidden layers must be positive')
   -- assert(#opt.numPos == #opt.numNeg, 'Provide right input format for number of positive and negative examples.')
   -- assert(opt.classHiddenUnits[1] == opt.nHiddenUnits[#opt.nHiddenUnits], 'Provide right input for number of hidden units.')

   local type = opt.weightInit
   if torch.type(type) ~= 'table' then
      opt.weightInit = {}
      for i = 1, #opt.nHiddenUnits + #opt.classHiddenUnits do   opt.weightInit[#opt.weightInit + 1] = type
      end
   end

   opt.numPos = {opt.numPos}
   opt.numNeg = {opt.numNeg}


   assert(#opt.weightInit == #opt.nHiddenUnits + #opt.classHiddenUnits, 'Provide right input format for weight initialization at each layer.')

   for k = 1, #opt.numPos do
      opt.numPos[k] = tonumber(opt.numPos[k])
      opt.numNeg[k] = tonumber(opt.numNeg[k])
   end

   if #opt.numPos > 1 then
      assert(#opt.numNeg == #opt.weightInit, 'Provide right input format for number of positive example with respect to number of hidden layers.')
   else
      for i = 2, #opt.weightInit do
        opt.numPos[i] = opt.numPos[1]
        opt.numNeg[i] = opt.numNeg[1]
      end
   end

   if #opt.numPoints > 1 then
      for k = 1, #opt.numPoints do
         opt.numPoints[k] = tonumber(opt.numPoints[k])
      end
   else
       for i = 2, #opt.weightInit do
        opt.numPoints[i] = tonumber(opt.numPoints[1])
       end
   end

end
