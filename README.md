# ITensorInfiniteMPS

This is a package for working with infinite MPS based on the [ITensors.jl](https://github.com/ITensor/ITensors.jl) library. The goal is to provide basic tools for infinite MPS that match the functionality that is available for finite MPS in ITensors.jl, for example gauging infinite MPS with `orthogonalize`, `InfiniteMPS + InfiniteMPS`, `InfiniteMPO * InfiniteMPS`, gate evolution, computing low-lying excited states with VUMPS, etc.

This is a work in progress. Here are some examples of the interface:
```julia
julia> using ITensors, ITensorInfiniteMPS

julia> s = siteinds("S=1/2", 3)
3-element Array{Index{Int64},1}:
 (dim=2|id=652|"S=1/2,Site,n=1")
 (dim=2|id=984|"S=1/2,Site,n=2")
 (dim=2|id=569|"S=1/2,Site,n=3")

julia> ψ = InfiniteMPS(s) # Infinite MPS with 3-site unit cell
InfiniteMPS
[1] IndexSet{3} (dim=1|id=317|"Link,c=0,l=3") (dim=2|id=652|"S=1/2,Site,c=1,n=1") (dim=1|id=77|"Link,c=1,l=1")
[2] IndexSet{3} (dim=1|id=77|"Link,c=1,l=1") (dim=2|id=984|"S=1/2,Site,c=1,n=2") (dim=1|id=868|"Link,c=1,l=2")
[3] IndexSet{3} (dim=1|id=868|"Link,c=1,l=2") (dim=2|id=569|"S=1/2,Site,c=1,n=3") (dim=1|id=317|"Link,c=1,l=3")


julia> ψ[2] == replacetags(ψ[5], "c=2" => "c=1") # Indexing outside of the unit cell gets tensors from other unit cells
true

julia> ψ₁ = ψ[1:3] # Create a finite MPS from the tensors of the first unit cell
MPS
[1] IndexSet{3} (dim=1|id=317|"Link,c=0,l=3") (dim=2|id=652|"S=1/2,Site,c=1,n=1") (dim=1|id=77|"Link,c=1,l=1")
[2] IndexSet{3} (dim=1|id=77|"Link,c=1,l=1") (dim=2|id=984|"S=1/2,Site,c=1,n=2") (dim=1|id=868|"Link,c=1,l=2")
[3] IndexSet{3} (dim=1|id=868|"Link,c=1,l=2") (dim=2|id=569|"S=1/2,Site,c=1,n=3") (dim=1|id=317|"Link,c=1,l=3")


julia> ψ₂ = ψ[4:6] # Create a finite MPS from the tensors of the second unit cell
MPS
[1] IndexSet{3} (dim=1|id=317|"Link,c=1,l=3") (dim=2|id=652|"S=1/2,Site,c=2,n=1") (dim=1|id=77|"Link,c=2,l=1")
[2] IndexSet{3} (dim=1|id=77|"Link,c=2,l=1") (dim=2|id=984|"S=1/2,Site,c=2,n=2") (dim=1|id=868|"Link,c=2,l=2")
[3] IndexSet{3} (dim=1|id=868|"Link,c=2,l=2") (dim=2|id=569|"S=1/2,Site,c=2,n=3") (dim=1|id=317|"Link,c=2,l=3")
```
Useful operations like gauging and optimization are in progress, so stay tuned!

