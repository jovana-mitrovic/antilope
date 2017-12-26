
-- Mean normalization within class
function normalize_within_class(data)

					local d = data.data
					local labels = data.labels
					local out = {}

					for a = 1, labels:max() do
						  ind = torch.range(1, labels:nElement())[labels:eq(a)]
								local dummy = d:index(1, ind:long()):clone()
								print(torch.mean(dummy, 1))
								out[a] = dummy - torch.mean(dummy, 1):expandAs(dummy)
					end

					return {data = torch.cat(out, 1), labels = labels}
end

-- [0,1] normalization



-- Norm normalization
function norm_normalization(data)

	    local d = data.data
					for i = 1, d:size(1) do
						   d[i] = d[i] / torch.norm(d[i])
					end

					return {data = d, labels = data.labels}
end
