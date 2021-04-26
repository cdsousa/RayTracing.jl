module RayTracing

# using Distributed
using LinearAlgebra
using Images
using GeometryBasics

include("vec.jl")
include("ray.jl")
include("camera.jl")

abstract type Hitable end
abstract type Material end

struct HitRecord{T}
    t::Float64
    p::Vec3{T}
    normal::Vec3{T}
    mat::Material
end

include("material/material.jl")

struct Sphere{T} <: Hitable
    center::Vec3{T}
    radius::T
    mat::Material
end

function hit(s::Sphere, r::Ray, tmin::Float64, tmax::Float64)::Option{HitRecord}
    oc = r.origin - s.center
    a = (r.direction ⋅ r.direction)
    b = (oc ⋅ r.direction)
    c = (oc ⋅ oc) - s.radius^2
    discriminant = b^2 - a*c
    if discriminant > 0
        temp = -(b + sqrt(discriminant)) / a
        if tmin < temp < tmax
            t = temp
            p = point(r, t)
            n = (p - s.center) / s.radius
            return HitRecord(t, p, n, s.mat)
        end
        temp = (-b + sqrt(discriminant)) / a
        if tmin < temp < tmax
            t = temp
            p = point(r, t)
            n = (p - s.center) / s.radius
            return HitRecord(t, p, n, s.mat)
        end
    end
    return missing
end


struct HitableList <: Hitable
    list::Vector{Hitable}
end

function hit(h::HitableList, r::Ray, tmin::Float64, tmax::Float64)::Option{HitRecord}
    closest = tmax
    rec = missing
    for el in h.list
        temprec = hit(el, r, tmin, closest)
        if !ismissing(temprec)
            rec = temprec
            closest = rec.t
        end
    end
    rec
end

struct Scatter{T}
    ray::Ray{T}
    attenuation::Vec3{T}
    reflect::Bool
end

function color(r::Ray, world::Hitable, depth::Int)::Vec3
    rec = hit(world, r, 0.0, typemax(Float64))
    if !ismissing(rec)
        s = scatter(rec.mat, r, rec)
        if s.reflect && depth < 20
            return s.attenuation .* color(s.ray, world, depth+1)
        else
            return Vec3(0.0, 0.0, 0.0)
        end
    else
        unit_direction = Vec3{Float64}(r.direction)
        t = 0.5 * (unit_direction[2] + 1.0)
        (1.0 - t) .* Vec3(1.0, 1.0, 1.0) .+ t.*Vec3(0.5, 0.7, 1.0)
    end
end


function scene()::Hitable
    n = 500
    spheres = Sphere[]
    push!(spheres, Sphere{Float64}(Vec3(0.0, -1000.0, 0.0), 1000, Lambert(Vec3(0.5, 0.5, 0.5))))
    for a=-11:11
        for b=-11:11
            choose_mat = rand()
            center = Vec3(a+0.9*rand(), 0.2, b+0.9*rand())
            if norm(center .- Vec3(4.0, 0.2, 0.0)) > 0.9
                if choose_mat < 0.8
                    push!(spheres, Sphere(center, 0.2, Lambert(Vec3(rand()*rand(), rand()*rand(), rand()*rand()))))
                elseif choose_mat < 0.95
                    push!(spheres, Sphere(center, 0.2, Metal(Vec3(0.5*(1+rand()), 0.5*(1+rand()), 0.5*(1+rand())), 0.5*rand())))
                else
                    push!(spheres, Sphere(center, 0.2, Dielectric(1.5)))
                end
            end
        end
    end

    push!(spheres, Sphere(Vec3(0.0, 1.0, 0.0), 1.0, Metal(Vec3(0.7, 0.6, 0.5), 0.0)))
    push!(spheres, Sphere(Vec3(-4.0, 1.0, 0.0), 1.0, Lambert(Vec3(0.4, 0.2, 0.1))))
    push!(spheres, Sphere(Vec3(4.0, 1.0, 0.0), 1.0, Dielectric(1.5)))
    HitableList(spheres)
end



function render(w::Hitable, c::Camera, nx=400, ny=200, ns=100)
    img = Matrix{RGB{Float64}}(undef, ny, nx)
    for i in 1:nx
        for j in reverse(1:ny)
            cols = Vector{Vec3}(undef, ns)
            for k in 1:ns
                r = Ray(c, (i+rand())/nx, (j+rand())/ny)
                cols[k] = RayTracing.color(r, w, 0)
            end
            col = sum(cols) ./ ns
            col = sqrt.(col)
            ir = col[1]
            ig = col[2]
            ib = col[3]
            img[ny-j+1, i] = RGB(ir,ig,ib)
        end
    end
    img
end

end # module

img = RayTracing.render(RayTracing.scene(), RayTracing.Camera(), 400, 200, 1)
display(img)

using FileIO
save("img.png", img)


