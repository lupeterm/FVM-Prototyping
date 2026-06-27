#import "@preview/parcio-slides:0.1.2": *
#show: parcio-theme

#let dark = rgb("#172033")
#let muted = rgb("#667085")
#let soft = rgb("#eef3fb")
#let rule = rgb("#d8dee9")

#let note(body) = rect(
  width: 100%,
  inset: 10pt,
  radius: 4pt,
  fill: soft,
  stroke: 0.6pt + rule,
)[#body]

#let chip(body) = box(
  inset: (x: 7pt, y: 4pt),
  radius: 3pt,
  fill: soft,
  stroke: 0.5pt + rule,
)[#text(size: 0.72em, fill: dark)[#body]]

#title-slide(
  title: "Investigating Matrix Assembly Strategies for a GPU Accelerated OpenFOAM",
  subtitle: "OpenFOAM Workshop '26",
  logo: none,
  author: (name: "Lukas Petermanne et al.", mail: "example@ovgu.de"),
  extra: [
    #set text(0.825em)
    Faculty of Computer Science\
    Otto von Guericke University Magdeburg

    #v(0.45cm)
    #grid(
      columns: (auto, auto, auto, auto),
      gutter: 8pt,
      chip([NeoN]),
      chip([face based]),
      chip([operator fusing]),
      chip([Julia JIT]),
    )
  ],
)

// Show presentation title in outline and highlight upcoming section.
#outline-slide(show-title: true, new-section: "Introduction")

#slide(title: "Motivation", new-section: "Introduction")[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.55cm,
    [
      #text(weight: "bold", fill: dark)[OpenFOAM evaluates eagerly]

      - The momentum equation is assembled from operator expressions.
      - Each operator creates its own matrix contribution.
      - Intermediate matrices repeat mesh traversal and coefficient writes.
      - The result is inefficient and even more memory bound than the final linear solve suggests.
    ],
    [
      #note[
        #text(weight: "bold")[Core opportunity]

        If compatible operators can be fused before assembly, we can evaluate the momentum equation with fewer passes over mesh and field data.
      ]
    ],
  )
]

#slide(title: "Motivation: NeoN and Julia")[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.55cm,
    [
      #text(weight: "bold", fill: dark)[NeoN as GPU backend]

      - NeoN introduces a GPU-oriented backend for OpenFOAM-style finite-volume assembly.
      - The backend has to preserve OpenFOAM's runtime configurability.
      - Scheme choice, field type, boundary behaviour, and mesh layout are often known only at runtime.
      - Encoding all combinations in C++ pushes the design toward many templates and complex dispatch paths.
    ],
    [
      #text(weight: "bold", fill: dark)[Why delegate assembly to Julia?]

      - Julia specialises kernels at JIT compile time from runtime values.
      - Assembly variants can be expressed directly without exploding C++ template code.
      - The same interface can test cell-based, face-based, fused, CPU, and GPU kernels.
      - Successful variants can later inform a production C++ path.
    ],
  )

  #v(0.35cm)
  #note[
    The presentation therefore compares assembly strategies, but the larger point is architectural: use Julia as the specialization layer between OpenFOAM's dynamic model and NeoN's performance-oriented backend.
  ]
]

#outline-slide(new-section: "Methods")

#slide(title: "Baseline: OpenFOAM Assembly", new-section: "Methods")[
  #grid(
    columns: (0.92fr, 1.08fr),
    gutter: 0.6cm,
    [
      #text(weight: "bold", fill: dark)[Current model]

      - Discretization operators assemble into sparse matrix coefficients.
      - Mesh topology drives owner/neighbour access patterns.
      - Boundary conditions and schemes introduce many small branches.
      - Generality is excellent; performance tuning is constrained by interface and data layout.
    ],
    [
      #image("figures/paper_assembly.svg", width: 100%)
    ],
  )
]

#slide(title: "Assembly Strategy Space")[
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 0.35cm,
    [
      #note[
        #text(weight: "bold")[Cell based]

        Loop over cells, gather adjacent faces, update diagonal and source terms locally.

        #v(0.15cm)
        #text(size: 0.82em, fill: muted)[Simple locality, less direct face reuse.]
      ]
    ],
    [
      #note[
        #text(weight: "bold")[Face based]

        Loop over internal faces, update owner and neighbour coefficients from one face contribution.

        #v(0.15cm)
        #text(size: 0.82em, fill: muted)[Natural finite-volume traversal, write contention in parallel.]
      ]
    ],
    [
      #note[
        #text(weight: "bold")[Global face based]

        Batch or reorder face work to expose wider parallelism and more regular memory access.

        #v(0.15cm)
        #text(size: 0.82em, fill: muted)[Better for accelerators, higher orchestration cost.]
      ]
    ],
  )

  #v(0.45cm)
  #note[
    The comparison is not only about absolute time. The relevant axes are memory traffic, write conflicts, sparse access regularity, boundary handling, and how much OpenFOAM state must cross the language boundary.
  ]
]

#slide(title: "C++ to Julia Delegation")[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.55cm,
    [
      #text(weight: "bold", fill: dark)[OpenFOAM side]

      - Owns solver lifecycle, mesh, fields, dictionaries, and boundary conditions.
      - Extracts compact views of topology and coefficient data.
      - Calls Julia kernels for selected operator assembly paths.
      - Receives assembled matrix arrays or operator contributions.
    ],
    [
      #text(weight: "bold", fill: dark)[Julia side]

      - Implements assembly variants with common input contracts.
      - Uses multiple dispatch for scheme and operator specialisation.
      - Experiments with threading, batching, and GPU execution.
      - Keeps benchmark code close to numerical kernels.
    ],
  )

  #v(0.4cm)
  #align(center)[
    #text(size: 0.82em, fill: muted)[C++ solver shell -> compact mesh/operator views -> Julia kernels -> sparse matrix coefficients]
  ]
]

#slide(title: "Operator Fusing")[
  #grid(
    columns: (0.95fr, 1.05fr),
    gutter: 0.55cm,
    [
      #text(weight: "bold", fill: dark)[Idea]

      Instead of assembling each operator independently, combine compatible operator passes before writing final matrix coefficients.

      #v(0.3cm)
      #text(weight: "bold", fill: dark)[Target]

      - Reduce repeated mesh traversal.
      - Reduce intermediate allocation.
      - Improve cache reuse for field and geometry data.
      - Preserve scheme-level composability where possible.
    ],
    [
      #image("variants/fused_all.svg", width: 100%)
    ],
  )
]

#outline-slide(new-section: "Results")

#slide(title: "Performance Dimensions", new-section: "Results")[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.55cm,
    [
      #text(weight: "bold", fill: dark)[Measurements]

      - Assembly wall time per case and per mesh size.
      - Scaling across CPU threads.
      - GPU kernel time and transfer overheads.
      - Memory bandwidth pressure.
      - Normalisation by cells, faces, and nonzeros.
    ],
    [
      #text(weight: "bold", fill: dark)[Interpretation]

      - A faster micro-kernel can lose at the C++/Julia boundary.
      - Fusing helps most when traversal dominates arithmetic.
      - GPU speedups depend on batching and regular write patterns.
      - Strategy choice may differ between scalar operators and coupled solver steps.
    ],
  )
]

#slide(title: "CPU Strategy Snapshot")[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.45cm,
    [
      #image("figures/comparison_CPU-Parallel_speedup.svg", width: 100%)
    ],
    [
      #image("figures/icoFoam-julia-speedup.svg", width: 100%)
    ],
  )

  #v(0.15cm)
  #text(size: 0.78em, fill: muted)[Replace this caption with the final benchmark interpretation once the OpenFOAM Workshop data set is frozen.]
]

#slide(title: "GPU and Interface Snapshot")[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.45cm,
    [
      #image("figures/gpu_interface_strategies.svg", width: 100%)
    ],
    [
      #image("figures/interface-facebased_GPU.svg", width: 100%)
    ],
  )

  #v(0.15cm)
  #text(size: 0.78em, fill: muted)[The key comparison is kernel throughput versus total delegated assembly time.]
]

#outline-slide(new-section: "Conclusion")

#slide(title: "Expected Takeaways", new-section: "Conclusion")[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.55cm,
    [
      #text(weight: "bold", fill: dark)[Technical]

      - Matrix assembly should be treated as a family of traversal strategies.
      - Julia makes this strategy space cheap to explore.
      - Operator fusing is a practical way to trade abstraction boundaries for less memory traffic.
      - Interface design decides whether kernel improvements survive integration.
    ],
    [
      #text(weight: "bold", fill: dark)[For OpenFOAM]

      - Delegation can support experimentation without replacing the solver stack.
      - Compact mesh and field views are the critical contract.
      - Best candidates are expensive, repeated, and structurally regular operator paths.
      - The approach can guide future C++ implementations even when Julia is only a prototype path.
    ],
  )
]

#slide(title: "Next Steps Before the Workshop")[
  - Finalise benchmark cases and hardware metadata.
  - Freeze the C++/Julia interface variant used for reported numbers.
  - Add one slide with minimal code showing the delegated call path.
  - Replace provisional captions with measured conclusions.
  - Prepare a short live demo or reproducible script for one matrix assembly variant.

  #v(0.45cm)
  #note[
    Discussion point: which parts of OpenFOAM matrix assembly are good candidates for a stable delegation interface, and which should remain entirely in C++?
  ]
]
