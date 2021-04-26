struct Ray{T}
    origin::Vec3{T}
    direction::Vec3{T}
end

function point(r::Ray, t::Float64)::Vec3{Float64}
    r.origin .+ t .* r.direction
end
