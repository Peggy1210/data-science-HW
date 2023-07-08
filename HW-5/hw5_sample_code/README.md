# Data Science HW5

## Environment
* Basic packages
    * pytorch 1.12.1
    * torchvision 0.13.1
    * torchaudio 0.12.1
    * cudatoolkit 11.3
* Other packages
    * numba 0.57.0
        - Aim to accelerate the calculation of bottleneck methods, e.g. `compute_new_a_hat_uv()` and `connected_after()`.
        - `@njit` enables compilation of the decorated function to be in no Python mode, which shortens the calculation time

## Run
* Run script
    ```
    python3 main.py --input_file target_nodes_list.txt --data_path ./data/data.pkl --model_path saved-models/gcn.pt --use_pgu
    ```
* The Google Colab source code is provided in `ds_hw5.ipynb`


## TODO
* `attacker.py`
    - Implementation of Nettack
    - Structure and direct attck only
* `main.py`
    - Load data from `data_path`
    - Train GCN model for the dataset
    - Setup attacker and perform attack

## References
* https://arxiv.org/abs/1805.07984
* https://arxiv.org/abs/2009.03488
* https://arxiv.org/abs/2005.06149