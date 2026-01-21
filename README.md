# FVM-Prototyping

**First Milestone:** Translating the [C++ Prototype](https://github.com/chihtaw/FVM-CFD-prototype ) into Julia ✔️

TODOs:
- Refactor Project structure yet again so that the assembly method is plug n' play and can be separately benchmarked ✔️
- stability ✔️ (probably :tm: )


# Initial Benchmarks

Cases: 
1. Lid-Driven Cavity S (1M x 1M Mesh)
2. Lid-Driven Cavity M (8M x 8M Mesh)
3. Windsor Body        (6M x 6M Mesh)

## Assembly Time
    
   ![benchmark](figures/assembly.svg)

## Assembly Time, Normed by #Cells (Matrix Diagonal)

   ![benchmark](figures/assembly_normed_ncells.svg)


## Memory Usage

   ![benchmark](figures/memory_julia.svg)
