module utils

export lowVarianceFilter, dummyEncode, dummyEncodeAll, minMaxScale, confusionMatrix, metrics 

using Statistics
using DataFrames

function lowVarianceFilter(X; t=0.001)
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

function dummyEncodeAll(X)
## Purpose: dummy encodes all String columns in a DataFrame
## Input: A DataFrame
## Output: A DataFrame
    df = DataFrame()
    for i ∈ 1:size(X, 2)
        if string(typeof(X[1, i])) ∉ ["Float16", "Float32", "Float64", "Int8", "Int16", "Int32", "Int64"]
            de = dummyEncode(X[!, i], names(X)[i])
            for c ∈ names(de)
                df[!, c] = de[!, c]
            end
        else
            df[!, names(X)[i]] = X[!, i]
        end
    end
    return df
end

function dummyEncode(x, name)
## Purpose: dummy encodes a given vector, with the first entry dropped 
## as a baseline.
## Input: vector of Strings and a name to assign subseqent columns
## Output: DataFrame
    u = unique(x)
    df = DataFrame()
    for i ∈ eachindex(u)
        classname = "'" * u[i] * "'"
        colname = "$name" * "." * "$classname"
        df[!, colname] = convert.(Int8, x .== u[i])
    end
    return df[!, (1:end) .!= 1]
end

function minMaxScale(x::Vector)::Vector
## Purpose: Scales a vector of inputs to between zero and one
## Input: a vector
## Output: a vector
    x = (x .- minimum(x))./(maximum(x) .- minimum(x))
    return x
end

function confusionMatrix(y, ŷ)::Matrix
## Purpose: Generates a confusion matrix for binary target variables
## Input: A vector of actuals and a vector of predictions
## Outpus: Returns a 2x2 matrix
    C = zeros(2, 2)
    for i ∈ 0:1
        for j ∈ 0:1
            C[i+1, j+1] = sum((y .== i) .& (ŷ .== j))
        end
    end
    return C
end

function metrics(C::Matrix)
## Purpose: Returns a tuple of standard performance metrics for binary classification using a confusion matrix
## Input: A 2x2 confusion matrix
## Output: A named tuple
    accuracy = (C[1, 1]+C[2, 2])/(C[1, 1]+C[1, 2]+C[2, 1]+C[2, 2])
    precision = C[1, 1]/(C[1, 1]+C[2, 1])
    recall = C[1, 1]/(C[1, 1]+C[1, 2])
    F1_score = 2*(precision*recall)/(precision+recall)
    return (Accuracy = accuracy, Precision = precision, Recall = recall, F1_Score = F1_score)
end

end # module
