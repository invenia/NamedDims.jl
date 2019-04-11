using Documenter, NamedDims

makedocs(;
    modules=[NamedDims],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/invenia/NamedDims.jl/blob/{commit}{path}#L{line}",
    sitename="NamedDims.jl",
    authors="Invenia Technical Computing Corporation",
    assets=[
        "assets/invenia.css",
        "assets/logo.png",
    ],
)

deploydocs(;
    repo="github.com/invenia/NamedDims.jl",
)
