# This .jl file is to illustrate a simple artifical neural network build using Julia's Flux package. No pre-processing is performed on the data aside from grouping all claims ≥ 1 as 'at least one claim'. The goal is to train a neural network that can reasonabilty predict whether a policyholder claimed or not
# Author: Patrick Moehrke
# Email: patrickmoehrke46@gmail.com


using Flux, CSV, DataFrames, MLJ, MLUtils
using MLJ: partition
using Flux: Dense, train!

## We load Australian private auto insurance claim data from 2004, sourced from R's 'CASdatasets' package. Original reference: P. De Jong and G.Z. Heller (2008), Generalized linear models for insurance data, Cambridge University Press.

data = DataFrame(CSV.File("../data/ausprivauto0405.csv"))

## ClaimOcc is the target variable, and we create a binary indicator from it and drop ClaimNb and ClaimAmount as features.
data = select!(data, Not(:ClaimNb));
data = select!(data, Not(:ClaimAmount));

## We apply min-max scaling to the VehValue
minmax(x) = (x .- minimum(x))./(maximum(x) .- minimum(x))
data[:, :VehValue] = minmax(data[:, :VehValue])

function dummy_encode(x, name)::DataFrame
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

function dummy_encode_all(X::DataFrame)::DataFrame
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

## We dummy encode our DataFrame and transform it into a matrix for Flux to interpret.
data_encode = Matrix{Float64}(dummy_encode_all(data));

X = data_encode[:, 1:(end-1)]
y = data_encode[:, end]


## By using partition, we ensure the distribution of out target variable is evenly dispersed
train, test = partition(eachindex(y), 0.7)

## TODO: Apply ROSE in order to address class imbalance in the training set
train_X, train_y = X[train, :], y[train, :] 
train_data = [(train_X', train_y')]

## Keep testing set imbalances like original data
test_X = X[test, :]
test_y = y[test, :]

## We construct an ANN with one hidden layer of 10 neurons. the tanh activation function scales inputs between -1 and 1 and introduces non-linearity
n_features = size(train_X, 2)
model = Chain(Dense(n_features, 10, tanh), Dense(10, 1, sigmoid))
β = Flux.params(model)

## Adam is chosen as the optimiser and MSE the loss function
opt = ADAM()
ℓ(x, y) = Flux.Losses.mse(model(x), y)

## We train using 2'000 epochs and display the loss every 200 epochs for transparency
println("Starting training...")
@show ℓ(train_X', train_y')

for epoch ∈ 1:2_000
    Flux.train!(ℓ, β, train_data, opt)
    if epoch % 200 == 0
        @show epoch
        @show ℓ(train_X', train_y')
    end
end

## Once trained, our model that generates the predictions assigns a 1 (at least one claim has occured) to all output values from the sigmoid output function greater than 0.5 and zero otherwise (no claim has occured)
predict(x) = model(x) .> 0.5

function confusion_matrix(y, ŷ)::Matrix
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

## Our final model can now be validated on unseen testing data
println("Predict on unseen data...")
CM = confusion_matrix(test_y', predict(test_X'))

@show metrics(CM)
display(CM)
println()
