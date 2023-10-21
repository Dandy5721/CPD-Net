# Detecting Brain State Changes by Geometric Deep Learning of Functional Dynamics on Riemannian Manifold

This repo contains the code of our MICCAI 2021 paper [Detecting Brain State Changes by Geometric Deep Learning of Functional Dynamics on Riemannian Manifold](https://link.springer.com/chapter/10.1007/978-3-030-87234-2_51).

## Usage

1. create an environment

```bash
conda env create -f requirements.yml
```

2. activate the environment

```bash
conda activate MICCAI-2021
```

3. run train.py

```bash
python train.py
```

## Reference

If you find our work useful in your research, please consider citing:

```bibtex
@InProceedings{10.1007/978-3-030-87234-2_51,
    author="Huang, Zhuobin and Cai, Hongmin and Dan, Tingting and Lin, Yi and Laurienti, Paul and Wu, Guorong",
    editor="de Bruijne, Marleen and Cattin, Philippe C. and Cotin, St{\'e}phane and Padoy, Nicolas and Speidel, Stefanie and Zheng, Yefeng and Essert, Caroline",
    title="Detecting Brain State Changes by Geometric Deep Learning of Functional Dynamics on Riemannian Manifold",
    booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2021",
    year="2021",
    publisher="Springer International Publishing",
    address="Cham",
    pages="543--552",
    isbn="978-3-030-87234-2"
}
```