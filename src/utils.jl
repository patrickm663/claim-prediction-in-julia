using Statistics
using DataFrames
using Optim

function LowVarianceFilter(X; t=0.001)
## Purpose: drops columns with low/zero variance
## Input: A Matrix (DataFrame untested), a threshhold t (default=0.001)
## Output: A Matrix (DataFrame untested)
    l = zeros(0)
    for c ∈ 1:size(X, 2)
        if var(X[:, c]) ≤ t
            append!(l, c)
        end
    end
    return X[:, (1:end) .∉ (l,)] 
end

function DummyEncodeAll(X::DataFrame)::DataFrame
## Purpose: dummy encodes all String columns in a DataFrame
## Input: A DataFrame
## Output: A DataFrame
    df = DataFrame()
    for i ∈ 1:size(X, 2)
        if string(typeof(data[1, i])) ∉ ["Float64", "Int64", "Int8", "Int16", "Int32"]
            de = dummy_encode(X[!, i], names(X)[i])
            for c ∈ names(de)
                df[!, c] = de[!, c]
            end
        else
            df[!, names(X)[i]] = X[!, i]
        end
    end
    return df
end

function DummyEncode(x, name)::DataFrame
## Purpose: dummy encodes a given vector, with the first entry dropped 
## as a baseline.
## Input: vector of Strings and a name to assign subseqent columns
## Output: DataFrame
    u = unique(x)
    df = DataFrame()
    for i ∈ eachindex(u)
        colname = "$name" * "$i"
        df[!, colname] = x .== u[i]
    end
    return df[!, (1:end) .!= 1]
end

function MinMaxScale(x::Vector)::Vector
## Purpose: Scales a vector of inputs to between zero and one
## Input: a vector
## Output: a vector
    x = (x .- minimum(x))./(maximum(x) .- minimum(x))
    return x
end

function YJ(y::Vector, λ)::Vector
## Purpose: Performs the Yeo-Johnson power transformation
## Input: a vector and λ
## Output: a vector
    for i in 1:length(y)
        if y[i] ≥ 0
            if λ ≠ 0
                y[i] = ((y[i] + 1)^λ - 1)/λ
            else
                y[i] = log(y[i] + 1)
            end
        else
            if λ ≠ 2
                y[i] = -((-y[i] + 1)^(2 - λ) - 1)/(2 - λ)
            else
                y[i] = -log(-y[i] + 1)
            end
        end
    end
    return y
end

function YeoJohnson(y::Vector; min = -10, max = 10, λ = 0.1, opt = true)::Vector
## Purpose: Calls the _YeoJohnson depending on whether lambda should be optimised or not
## Input: A Vector. Optional inputs are the min and max search range for the optimiser, an optional λ parameter, and a flag for whether to optimise or use the λ input provided
## Output: a Vector of Yeo-Johnson transformed values
    if opt == true
        return YJ(y, λoptimimum(y, min, max))
    else
        return YJ(y, λ)
    end
end

function LogLike(y, λ)::Float32
## Purpose: Computes the log-likelihood of the Yeo-Johnson power transformation
## Input: a Vector and λ parameter
## Output: a Float
## Source: Algorithm from SciPy.stats `yeojohnson_lif` function in _morestats.py
    N = length(y)
    σ̂² = var(YJ(y, λ))
    LL = -N/2 * log(σ̂²) + (λ - 1) * sum(sgn.(y) .* log.(abs.(y) .+ 1))
    return LL
end

function sgn(x)
## Purpose: Calculates the "sign" of a number (1 if positive, -1 if negative, zero otherwise)
## Input: a number
## Output: 1, -1, or 0
    if x ≠ 0.0
        return x/abs(x)
    else
        return 0
    end
end

function λoptimimum(y::Vector, min, max)::Float32
## Purpose: Computes the value of λ that minimises the log-likelihood of the Yeo-Johnson power transformation
## Input: a Vector and minimum and maximum values for the search range
## Output: a Value for λ̂, the minimiser
    LL(λ) = LogLike(y, λ)
    λ̂ = optimize(LL, min, max).minimizer
    return λ̂
end
