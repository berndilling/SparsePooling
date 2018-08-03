
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
end

mutable struct Image
  image::Array{Float64, 2}
end

##############################################################
# Constructors

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

####### Rendering

@inline function renderobject!(object::Bar, image::Image)
  object.orientation_horizontal ?
  image.image[object.position[1]:object.position[1]+object.width-1,
    object.position[2]:object.position[2]+object.length-1] = 1 :
  image.image[object.position[1]:object.position[1]+object.length-1,
    object.position[2]:object.position[2]+object.width-1] = 1
end
@inline function renderobject!(object::Square, image::Image)
  for edge in object.edges
    renderobject!(edge, image)
  end
end
@inline function renderobject!(object::CompositeObject, image::Image; rand_pos = true)
  for comp in object.components
    renderobject!(comp, image)
  end
  rand_pos && (image.image = circshift(image.image,rand(0:size(image.image)[1],2)))
end

######### Moving: Generatorfunctions for SparsePooling learning function

@inline function getobject(; image_size = 32, n_of_components = rand(1:4))
  image = Image(zeros(image_size,image_size))
  renderobject!(generatecompositeobject(n_of_components), image)
  return image
end
@inline function getmovingobject(image::Image; duration = 20, background = [], speed = 1)
   sequence = zeros(size(image.image)[1], size(image.image)[2], duration)
   direction = rand([[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]])
   for i in 1:duration
      sequence[:,:,i] = circshift(image.image,i*speed*direction)
   end
   !isempty(background) && [sequence[:,:,i] =
    clamp.(sequence[:,:,i] + background,0,1) for i in 1:duration]
   return sequence
end
@inline getstaticobject(image::Image) = reshape(image.image, size(image.image)[1],
                                                  size(image.image)[1],1)

############################################################
# Testing

# using PyPlot
#  close("all")
# object = generatecompositeobject(3)
# image = Image(zeros(32,32))
# renderobject!(object, image)
# imshow(image.image)

# image2 = Image(zeros(32,32))
# figure()
# renderobject!(generateline(),image2)#generatepyramid()
# imshow(image2.image)
#
# sequence = getmovingobject(image; duration = 20, speed = 2, background = image2.image)#get_background())
# print(size(sequence))
# figure()
# for i in 1:size(sequence)[3]
#   imshow(sequence[:,:,i])
#   sleep(0.1)
# end

# image = getobject()
# dynamicimage = getmovingobject(image)
# print(size(dynamicimage))
# figure()
# for i in 1:size(dynamicimage)[3]
#   imshow(dynamicimage[:,:,i])
#   sleep(0.1)
# end
