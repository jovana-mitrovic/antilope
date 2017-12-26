require 'math'
c = require 'trepl.colorize'

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
