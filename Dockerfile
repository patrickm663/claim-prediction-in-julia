# Author: Patrick Moehrke
# Licence: MIT

## Using Ubuntu 22.10 as a base, install Pip via APT
FROM ubuntu:22.10
RUN apt update && apt upgrade
RUN apt install pip git wget -y

## With Pip installed, install JupyterLab
RUN pip install --upgrade pip
RUN pip install jupyterlab

## Install Julia and IJulia
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.2-linux-x86_64.tar.gz
RUN tar -xvzf julia-1.8.2-linux-x86_64.tar.gz
RUN cp -r julia-1.8.2 /opt/
RUN ln -s /opt/julia-1.8.2/bin/julia /usr/local/bin/julia
RUN rm julia-1.8.2-linux-x86_64.tar.gz
RUN julia -e 'using Pkg; Pkg.add("IJulia"); Pkg.build("IJulia")'
EXPOSE 8888

## Copy the repo's files and data into the container (see .dockerignore for exlusions)
WORKDIR /notebook
COPY . .

## Activate the project and install dependencies
RUN julia -e 'using Pkg; Pkg.activate("src/"); Pkg.instantiate()'

## Start JupyterLab on boot
CMD jupyter lab --allow-root --ip 0.0.0.0 --port 8888 
