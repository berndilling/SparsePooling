#using StatsBase, PyPlot

type PatternParameter
  edge_length::Int64 # linear dimension of complete field of view
  pattern_duration::Int64 # number of timesteps per pattern
  number_of_bars::Array{Int64, 1} # equally distributed?
  weights_n_of_bars::Array{Float64, 1} # weights for number of bars
  bar_lengths::Array{Int64, 1}
  bar_widths::Array{Int64, 1}
  bar_orientations::Array{Array{Int64, 1}, 1} # [1,0]:horizontal, [0,1]:vertical
  connections::Array{Array{Int64, 1}, 1} # all uneven number up to bar length
  directions::Array{Array{Int64, 1}, 1}
end
function PatternParameter(;
  edge_length = 32,
  pattern_duration = 20,
  number_of_bars = [1,2,3,4,5],
  weights_n_of_bars = [1,1,1,1,1]/5,
  bar_lengths = [5,7,9], #should not be smaller than 5!
  bar_widths = [1],
  bar_orientations = [[1,0],[0,1]],
  connections = [[2*j-1 for j in 1:Int(ceil(i/2))] for i in [5,7,9]],
  directions = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]] #[[1,0],[0,1],[-1,0],[0,-1]]
  )
  PatternParameter(edge_length, pattern_duration, number_of_bars, weights_n_of_bars, bar_lengths, bar_widths,
                    bar_orientations, connections, directions)
end

#function _setfirstbar()

#function _setbar()

function get_pattern(parameter = PatternParameter())
    pattern = zeros(parameter.edge_length,parameter.edge_length)
  n_of_bars = sample(parameter.number_of_bars,Weights(parameter.weights_n_of_bars))
  index = rand(1:length(parameter.bar_lengths))
  for i in 2:n_of_bars
    index = rand(1:length(parameter.bar_lengths))
    bar_length = parameter.bar_lengths[index]
    position = rand(maximum(parameter.bar_lengths):parameter.edge_length-maximum(parameter.bar_lengths),2)
    orientation = rand(parameter.bar_orientations)
    pattern[position[1]:position[1]+bar_length*orientation[1],
            position[2]:position[2]+bar_length*orientation[2]] = 1
  end
  return pattern
end

#argument "pattern" must be matrix of ZEROS with edge_length = parameter.edge_length
function get_connected_pattern(parameter = PatternParameter())
  pattern = zeros(parameter.edge_length,parameter.edge_length)
  n_of_bars = sample(parameter.number_of_bars,Weights(parameter.weights_n_of_bars))
  index = rand(1:length(parameter.bar_lengths))
  bar_length = parameter.bar_lengths[index]
  position = rand(maximum(parameter.bar_lengths)+1:parameter.edge_length-maximum(parameter.bar_lengths)-1,2)
  orientation = rand(parameter.bar_orientations)
  pattern[position[1]:position[1]+bar_length*orientation[1],
          position[2]:position[2]+bar_length*orientation[2]] = 1
  for i in 2:n_of_bars
    connection_1 = rand(parameter.connections[index])
    index = rand(1:length(parameter.bar_lengths))
    bar_length = parameter.bar_lengths[index]
    connection_2 = rand(parameter.connections[index])
    position .+= [orientation[1] -orientation[2]; orientation[2] -orientation[1]]*[connection_1,connection_2]
    orientation = 1.-orientation #change orientation for every subsequent bar
    pattern[clamp.(position[1]:position[1]+bar_length*orientation[1],1,parameter.edge_length),
            clamp.(position[2]:position[2]+bar_length*orientation[2],1,parameter.edge_length)] = 1
  end
  return pattern
end

function get_background()
  return get_pattern(parameter = PatternParameter(number_of_bars = [12], weights_n_of_bars = [1.]))
end

function get_moving_pattern(pattern::Array{Float64, 2}, parameter; background = [])
   pattern_sequence = zeros(size(pattern)[1],size(pattern)[2],parameter.pattern_duration)
   direction = rand(parameter.directions)
   for i in 1:parameter.pattern_duration
      pattern_sequence[:,:,i] = circshift(pattern,i*direction)
   end
   !isempty(background) && [pattern_sequence[:,:,i] = clamp.(pattern_sequence[:,:,i]+background,0,1) for i in 1:parameter.pattern_duration]
   return pattern_sequence
end
#return image patches for training patchy sparse layer
function cut_pattern(pattern;
  full_edge_length = 32,
  patch_edge_length = 8,
  overlap = 4)
  number_of_patches = Int(32/(patch_edge_length-overlap)-1)
  patches = zeros(patch_edge_length,patch_edge_length,number_of_patches^2)
  for i in 1:number_of_patches
    for j in 1:number_of_patches
      patches[:,:,(i-1)*number_of_patches+j] =
      pattern[(i-1)*(patch_edge_length-overlap)+1:i*(patch_edge_length)-(i-1)*overlap,
              (j-1)*(patch_edge_length-overlap)+1:j*(patch_edge_length)-(j-1)*overlap]
    end
  end
  return patches
end

#pattern = get_connected_pattern()
#pattern = get_pattern()
#imshow(pattern)

# figure()
# background = zeros(32,32)
# imshow(get_background!(background))
#

# pattern = get_connected_pattern!()
# pattern_seq = get_moving_pattern(pattern, PatternParameter(pattern_duration = 10), background = get_pattern!())
# for i in 1:10
#   figure()
#   imshow(pattern_seq[:,:,i])
# end
