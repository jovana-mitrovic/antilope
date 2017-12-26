require 'lfs'
require 'gnuplot'
require 'utils.aux_fct'

-- dataset = 'xor'
-- directory = '/Users/jovana/Desktop/nips_project/torch/' .. dataset .. '/logs/1000-100-2/overlap*0.5dim*998noise*0.5'
dataset = 'mnist'
directory = '/Users/jovana/Desktop/nips_project/kernel_init/'
             .. dataset .. '/logs/784-800-10/adam/*lr*0.001*mom*0.9*lrd*1e-07*L1*0*L2*0/bn+drop/no_rescale'


settings = {}
_, max_depth = string.gsub(directory, '/', '')
file_length = 0

for filename, attr in dirtree(directory) do
    _, depth = string.gsub(filename, '/', '')
    if attr.mode == 'directory' and depth > max_depth then
        max_depth = depth
    end

    if attr.mode == 'file' then
       local l = -1
       for _ in io.lines(filename) do l = l + 1 end
       file_length = math.max(file_length, l)
    end
end

for filename, attr in dirtree(directory) do
    _, depth = string.gsub(filename, '/', '')
    if attr.mode == 'directory' and depth == max_depth then
       settings[#settings + 1] = filename:sub(#directory + 1)
    end
end


if #settings == 0 then settings[1] = '' end
num_files = torch.Tensor(#settings):zero()
file_list = {}

-- Counting the number of files in the deepest subdirectory
for i = 1, #settings do
   local files = {}
   for filename, attr in dirtree(directory) do
      if attr.mode == 'file' then
         if string.find(filename, settings[i], 1, true) then
            num_files[i] = num_files[i] + 1
            files[#files + 1] = filename
         end
      end
   end
   file_list[#file_list + 1] = files
end



skip_first_lines = 1
skip_last_lines = 0
max_lines = 50
file_length = math.min(file_length - skip_first_lines, max_lines - skip_first_lines)

print(file_list)


for i = 1, #file_list do

   -- Get keys
   file = io.open(file_list[i][1])
   for line in file:lines() do
       keys = line:split('\t')
       break
   end

   -- Tensor to hold txt file data
    local res = {}
    for _, val in pairs(keys) do
       res[val] = torch.Tensor(num_files[i], file_length):zero()
    end


    local types = {}
    -- iterate over files
    for j = 1, #file_list[i] do
        types[#types + 1] = file_list[i][j]:sub(#directory + #settings[i] + 2):gsub('%.txt', '')
        types[#types] = types[#types]:split('_')[1]
        file = io.open(file_list[i][j])

        -- skip first line
        for q = 1, skip_first_lines do
           for line in file:lines() do
                break
           end
        end

        local p = 1
        for line in file:lines() do
          local values = line:split(' ')
          for t, val in pairs(values) do
              if pcall(function() res[keys[t]][{{j}, {p}}] = val end) then
                 res[keys[t]][{{j}, {p}}] = val
              else
                 b = val:split('\t')
                 res[keys[t]][{{j}, {p}}] = b[1]
                 res[keys[t + 1]][{{j}, {p}}] = b[2]
              end
          end

          p = p + 1
          if p > max_lines - skip_last_lines - 1 then break end
        end
    end


    -- function paste(t)
    --  local res = t[1]
    --   for i = 2, #t-2 do
    --         res = res .. '-' .. t[i]
    --    end
    --    return res
    -- end
    --
    --
    -- for kk = 1, #types do
    --     local tt = types[kk]:split('-')
    --     types[kk] = paste(tt)
    -- end

    for t, val in pairs(res) do
        plotSettings = {}
        for j = 1, #types do
            if string.find(types[j], 'N') == nil then mark = '~' else mark = '*' end
            plotSettings[#plotSettings + 1] = {types[j], val[j], mark}
        end

        name = directory .. '/plots' .. settings[i] .. ' ' .. t .. '.png'
        -- print(name)
        -- gnuplot.pngfigure(name)
        gnuplot.figure()
        gnuplot.title(settings[i] .. ' ' .. t)
        gnuplot.plot(plotSettings)
        gnuplot.movelegend('right', 'bottom')
        gnuplot.xlabel('Epochs')
        gnuplot.ylabel('Classification accuracy')
        -- gnuplot.plotflush()
    end
end
