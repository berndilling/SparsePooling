
using StatsBase

##############################################################################
## Types

abstract type Object end

struct Bar <: Object
  position::Array{Int64, 1}
  length::Int64
  width::Int64
  orientation_horizontal::Bool
end
struct Square <: Object
  position::Array{Int64, 1}
  edges::Array{Bar, 1}
end
struct Rectangle <: Object
  position::Array{Int64, 1}
  edges::Array{Bar, 1}
end

struct CompositeObject <: Object
  position::Array{Int64, 1}
  nr_components::Int64
  components::Array{Square, 1}
  anchor::Array{Int64, 1}
  object_dims::Array{Int64, 1}
end
@inline function CompositeObject(position, nr_components, components)
  anchor = [minimum([comp.position[1] for comp in components]),minimum([comp.position[2] for comp in components])]
  CompositeObject(position, nr_components, components, anchor,
  [maximum([comp.position[1] for comp in components]) + components[1].edges[1].length - 1 - anchor[1],
  maximum([comp.position[2] for comp in components]) + components[1].edges[1].length - 1- anchor[2]])
end

mutable struct Image
  image::Array{Float64, 2}
end
mutable struct AnchoredImage
  image::Array{Float64, 2}
  anchor::Array{Int64, 1}
  object_dims::Array{Int64, 1}
  anchorboundaries::Array{Int64, 1}
end
@inline function AnchoredImage(image::Array{Float64, 2})
  AnchoredImage(image, [0,0], [0,0], [0,0])
end

##############################################################################
## Constructors

@inline function generatebar(pos::Array{Int64, 1}, l::Int64, w::Int64, or::Bool)
  Bar(pos, l, w, or)
end
@inline function generatesquare(pos::Array{Int64, 1}; edgelength = rand([5,7,9]), edgewidth = rand([1,2]))
  Square(pos, [generatebar(pos,edgelength,edgewidth,true),generatebar(pos,edgelength,edgewidth,false),
    generatebar(pos+[0,edgelength-edgewidth],edgelength,edgewidth,false),
    generatebar(pos+[edgelength-edgewidth,0],edgelength,edgewidth,true)])
end
@inline function generatecompositeobject(nr_comp::Int64; pos = [16,16])
  CompositeObject(pos, nr_comp,
    [generatesquare(pos + rand(-6:6,2); edgelength = 9, edgewidth = 1) for i in 1:nr_comp])
end
@inline function generatepyramid(; pos = [16,16])
  CompositeObject(pos, 3,
    [generatesquare(pos - [0,3]; edgelength = 7, edgewidth = 1),
    generatesquare(pos + [0,3]; edgelength = 7, edgewidth = 1),
    generatesquare(pos - [6,0]; edgelength = 7, edgewidth = 1)
    ])
end
@inline function generatesupersquare(; pos = [16,16])
  CompositeObject(pos, 4,
    [generatesquare(pos - [3,3]; edgelength = 7, edgewidth = 1),
    generatesquare(pos + [3,3]; edgelength = 7, edgewidth = 1),
    generatesquare(pos + [3,-3]; edgelength = 7, edgewidth = 1),
    generatesquare(pos + [-3,3]; edgelength = 7, edgewidth = 1)
    ])
end
@inline function generateline(; pos = [16,16])
  CompositeObject(pos, 4,
    [generatesquare(pos - [0,3]; edgelength = 7, edgewidth = 1),
    generatesquare(pos + [0,3]; edgelength = 7, edgewidth = 1),
    generatesquare(pos - [0,9]; edgelength = 7, edgewidth = 1),
    generatesquare(pos + [0,9]; edgelength = 7, edgewidth = 1)
    ])
end

##############################################################################
## Tetris objects

@inline function tetris1(pos)
  CompositeObject(pos, 3,
    [generatesquare(pos - [4,4]; edgelength = 9, edgewidth = 1),
    generatesquare(pos - [4,12]; edgelength = 9, edgewidth = 1),
    generatesquare(pos + [-4,4]; edgelength = 9, edgewidth = 1)])
end
@inline function tetris2(pos)
  CompositeObject(pos, 3,
    [generatesquare(pos - [4,3]; edgelength = 9, edgewidth = 1),
    generatesquare(pos - [12,3]; edgelength = 9, edgewidth = 1),
    generatesquare(pos - [-4,3]; edgelength = 9, edgewidth = 1)])
end
@inline function tetris3(pos)
CompositeObject(pos, 3,
  [generatesquare(pos - [8,8]; edgelength = 9, edgewidth = 1),
  generatesquare(pos - [8,0]; edgelength = 9, edgewidth = 1),
  generatesquare(pos - [0,8]; edgelength = 9, edgewidth = 1)])
end
@inline function tetris4(pos)
  CompositeObject(pos, 2,
    [generatesquare(pos - [8,8]; edgelength = 9, edgewidth = 1),
    generatesquare(pos - [8,0]; edgelength = 9, edgewidth = 1),
    generatesquare(pos - [0,0]; edgelength = 9, edgewidth = 1)])
end
@inline function tetris5(pos)
  CompositeObject(pos, 2,
    [generatesquare(pos - [8,8]; edgelength = 9, edgewidth = 1),
    generatesquare(pos - [0,8]; edgelength = 9, edgewidth = 1),
    generatesquare(pos - [0,0]; edgelength = 9, edgewidth = 1)])
end
@inline function tetris6(pos)
  CompositeObject(pos, 2,
    [generatesquare(pos - [0,8]; edgelength = 9, edgewidth = 1),
    generatesquare(pos - [8,0]; edgelength = 9, edgewidth = 1),
    generatesquare(pos - [0,0]; edgelength = 9, edgewidth = 1)])
end
@inline function generatetetris(; pos = [16,16])
  rand([tetris1(pos),tetris2(pos),tetris3(pos),tetris4(pos),tetris5(pos),tetris6(pos)])
end

##############################################################################
## Rendering

@inline function renderobject!(object::Bar, image)
  object.orientation_horizontal ?
  image.image[object.position[1]:object.position[1]+object.width-1,
    object.position[2]:object.position[2]+object.length-1] = 1 :
  image.image[object.position[1]:object.position[1]+object.length-1,
    object.position[2]:object.position[2]+object.width-1] = 1
end
@inline function renderobject!(object::Square, image)
  for edge in object.edges
    renderobject!(edge, image)
  end
end
@inline function renderobject!(object::CompositeObject, image; rand_pos = true)
  for comp in object.components
    renderobject!(comp, image)
  end
  rand_pos && (image.image = circshift(image.image,rand(0:size(image.image)[1],2)))
end

##############################################################################
## Generatorfunctions for SparsePooling learning function

@inline function getbar(; image_size = 32, w = 1, or = rand([true,false]))
  pos = rand(1:image_size,2)
  l = image_size
  or ? (pos[2] = 1) : (pos[1] = 1)
  image = Image(zeros(image_size,image_size))
  renderobject!(generatebar(pos, l, w, or), image)
  return image
end
@inline function getobject(; image_size = 32, n_of_components = rand(1:4))
  image = Image(zeros(image_size,image_size))
  renderobject!(generatecompositeobject(n_of_components), image)
  return image
end
# TAKE CARE: anchors/boundaries only work if all atoms have same edge length!!!
@inline function getanchoredobject(; image_size = 32)
  image = AnchoredImage(zeros(image_size,image_size))
  object = sample([generatecompositeobject(3),generatetetris()],Weights([0.,1.]))
  image.anchor = deepcopy(object.anchor)
  image.object_dims = deepcopy(object.object_dims)
  image.anchorboundaries = [image_size - image.object_dims[1],image_size - image.object_dims[2]]
  renderobject!(object, image; rand_pos = false)
  return image
end

##############################################################################
## Dynamic functions

@inline function getmovingobject(image; duration = 8, background = [], speed = 1)
   sequence = zeros(size(image.image)[1], size(image.image)[2], duration)
   direction = rand([[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]])
   for i in 1:duration
      sequence[:,:,i] = circshift(image.image,i*speed*direction)
   end
   !isempty(background) && [sequence[:,:,i] =
    clamp.(sequence[:,:,i] + background,0,1) for i in 1:duration]
   return sequence
end
@inline function getjitteredobject(image; duration = 8, background = [])
  sequence = zeros(size(image.image)[1], size(image.image)[2], duration)
  for i in 1:duration
    direction = rand([[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]])
    distance = rand(0:div(size(image.image)[1],4))
    sequence[:,:,i] = circshift(image.image, distance .* direction)
  end
  !isempty(background) && [sequence[:,:,i] =
   clamp.(sequence[:,:,i] + background,0,1) for i in 1:duration]
  return sequence
end
@inline function getbouncingobject(image::AnchoredImage; duration = 8, background = [], speed = 1)
  sequence = zeros(size(image.image)[1], size(image.image)[2], duration)
  directions = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]]
  dir = rand(directions)
  for i in 1:duration
    anchortest = image.anchor .+ speed .* dir
    if anchortest[1] >= image.anchorboundaries[1] || anchortest[2] >= image.anchorboundaries[2] ||
        anchortest[1] <= 1 || anchortest[2] <= 1
      dir[anchortest .> image.anchorboundaries] *= -1
      dir[anchortest .< 1] *= -1
    end
      image.image = circshift(image.image,speed*dir)
      sequence[:,:,i] = deepcopy(image.image)
      image.anchor = image.anchor .+ speed .* dir
  end
  !isempty(background) && [sequence[:,:,i] =
   clamp.(sequence[:,:,i] + background,0,1) for i in 1:duration]
  return sequence
end
@inline getstaticobject(image) = reshape(image.image, size(image.image)[1],
                                                  size(image.image)[1],1)

############################################################
# Testing
#
# using PyPlot
# close("all")
# object = generatecompositeobject(3)
# image = Image(zeros(32,32))
# renderobject!(object, image)
# imshow(image.image)

# image2 = getanchoredobject()
# figure()
# imshow(image2.image, origin = "lower")
#
# dynamicimage = getmovingobject(image2)# getbouncingobject(image2)
# print(size(dynamicimage))
# figure()
# for i in 1:size(dynamicimage)[3]
#  imshow(dynamicimage[:,:,i])
#  sleep(0.2)
# end

# sequence = getmovingobject(image; duration = 20, speed = 2, background = image2.image)#get_background())
# print(size(sequence))
# figure()
# for i in 1:size(sequence)[3]
#   imshow(sequence[:,:,i])
#   sleep(0.1)
# end
#
# sequence = getjitteredobject(image; duration = 20, background = [])#get_background())
# print(size(sequence))
# figure()
# for i in 1:size(sequence)[3]
#   imshow(sequence[:,:,i])
#   sleep(0.2)
# end

# figure()
# image = getbar()
# imshow(image.image)
# dynamicimage = getmovingobject(image)
# print(size(dynamicimage))
# figure()
# for i in 1:size(dynamicimage)[3]
#   imshow(dynamicimage[:,:,i])
#   sleep(0.1)
# end
