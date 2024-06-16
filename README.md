
## Project Structure


```
BIPEFT/
│
├── examples_seq2seq/                     # datasets and data processing
├── space/                    # Directory for constructing search space
│   ├── __init__.py          # Makes src a Python module
│   ├── t5_search_space.py   # Contains the main modules for our BIPEFT design
│   ├── peft_modules.py      # Mixture of modules with diverse PEFT from S2 and S2
│   ├── peft_layers.py       # Some sub-modules of peft_modules
│   └── t5_forward_mom.py    # Modify the t5 forward functions for search
│
├── gumbel_module/           # Architecture weights forward processing, including gumbel_softmax
│
├── utils/                   # Util functions
├── scripts/                 # All the scripts for our experiments
│   ├── ablation             # Ablation study
│   ├── adaptation           # Generalization ability test
│   ├── budget               # Test on different budget levels
│   └── T5_searchs           # Main experiments
│
├── architect.py             # Differential NAS with first-order approximation
├── engine.py                # Engines for training and evaluation
└── train.py                 # Training launch file
```

## Installation

Instructions on setting up the project environment:

```bash
# For the python version, we use python=3.9

# Firstly, install pytorch based on your cuda version, for example we install the pytorch==2.2 with cuda toolkit 12.1 on Linux
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Install other required packages
pip install -r requirements.txt
```

## Usage

Run any script from ./scripts
```bash
# For example, to train the model with early stop and a default budget as 1.39%% on setting S1 and GLUE
./scripts/T5_searchs/S1/budget/search_es_1.39.sh

# Example of albation study: not using iterative search
./scripts/no_iter.sh
```
