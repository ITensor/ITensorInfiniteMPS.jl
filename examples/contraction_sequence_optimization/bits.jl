using ITensorsInfiniteMPS
using ITensorsInfiniteMPS.ContractionSequenceOptimization

import ITensorsInfiniteMPS.ContractionSequenceOptimization:
  _set

function get_first_nonzero_bit(i::Unsigned)
  n = 0
  @inbounds while !iszero(i)
    if isodd(i)
      return n+1
    end
    i = i >> 1
    n += 1
  end
  return n
end

is = UInt64[0, 1, 2, 4, 8, 16]
for i in is
  @show _set(i)
  @show get_first_nonzero_bit(i)
end

