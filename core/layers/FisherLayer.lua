require 'nn'

local FisherLayer, parent = torch.class('nn.FisherLayer', 'nn.Linear')

    function FisherLayer:__init(inputSize, outputSize, projData, schedule)

        assert(torch.type(projData) == 'table', 'Provide projection points as table with entries corresponding to output neurons.')
        assert(torch.type(schedule) == 'table', 'Provide schedule as table of indices.')

        parent.__init(self, inputSize, outputSize)
        torch.setdefaulttensortype(projData.data[1]:type())

        begin_index = 1

        for i = 1, #schedule.indices do
           local input = {}; ind = schedule.indices[i]:long()


           -- Select data corresponding to schedule entry
           input = {data = projData.data[i], labels = projData.labels[i]}
           input.labels = input.labels:reshape(input.labels:size(1), 1)

           -- Initialize data structures for calculating the Fisher LDA
           nClasses = input.labels:max()
           n, m = input.data:size(1), input.data:size(2)
           means = torch.Tensor(nClasses, m):zero()
           num = torch.Tensor(nClasses):zero()
           inClassScatter = torch.Tensor(m, m):zero()
           betwClassScatter = torch.Tensor(m, m):zero()

           -- Iterate through classes to compute class-wise means and in-class scatter matrix
           for a = 1, nClasses do
               local q = input.labels:eq(a); num[a] = q:sum()
               local qq = torch.nonzero(q)[{{},{1}}]
               qq = qq:reshape(qq:size(1))

               local dat = input.data:index(1, qq)
               means[a] = torch.mean(dat, 1)
               for b = 1, dat:size(1) do
                   local c = dat[b] - means[a]
                   inClassScatter:addr(c, c)
               end
            end

           -- Compute between class scatter matrix
           mean = torch.mean(means, 1)  -- mean of class-wise means
           for a = 1, nClasses do
               betwClassScatter:addr(1, num[a], means[a] - mean, means[a] - mean)
           end

           -- SVD approximation for inverse of inClassScatter matrix (for stability)
           u, s, v = torch.svd(inClassScatter)
           S = torch.diag(s[s:ge(1e-7)])
           U = u[{{}, {1, S:size(1)}}]
           V = v[{{}, {1, S:size(1)}}]
           inClassScatter_inverse = U * S * V:t()

           -- Get discriminant directions from the Fisher LDA
           res = inClassScatter_inverse * betwClassScatter
           e, W = torch.eig(res, 'V')
           _, indices = e[{{}, {1}}]:sort(1, true)
           indices = indices[{{1, nClasses - 1}, {}}]:reshape(nClasses - 1)
           mat = W:index(2, indices):t()  -- discriminant directions

           -- Fill weight matrix with discriminant directions
           local end_index = math.min(begin_index + mat:size(1) - 1, self.weight:size(1))
           local num_directions = end_index - begin_index + 1

           self.weight[{{begin_index, end_index}, {}}] = mat[{{1, num_directions}, {}}]

           -- Check if weight matrix is full or more discriminant directions need to be added
           begin_index = begin_index + mat:size(1)
           if begin_index > self.weight:size(1) then break end
       end
    end
