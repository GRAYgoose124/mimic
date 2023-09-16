# Mimic(net)
    Naive python implementations of neural nets and components. Goal: To rebuild various pieces from scratch in a composable way.

## Installation
```bash
# after cloning the repo
cd mimic
poetry install
```

## Usage
```bash
# Will demonstrate the model learning the xor function
mimic
```

## Features

- Models
  - [x] basic sequential model
  - [x] model saving/loading
    - [ ] w/ metadata
  - [x] generic activation/error functions
    - [ ] vectorized
  - [x] simple trainer
    - [ ] configurable backprop
  - [x] simple model visualization
    - [ ] better input/hidden/output display
- Datasets
  - [ ] mnist dataset
  - [x] dataset saving/loading


![network](https://github.com/GRAYgoose124/mimic/blob/main/network.png)

