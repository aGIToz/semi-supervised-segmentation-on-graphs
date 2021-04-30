| Image     | Seeds| Segmented |
| ----------- | ----------- | ----------- |
|<img src="./images/chien_tapestry_bayeux.png" alt="org_img" width="256" height="256">   | <img src="./images/seeds.png" alt="seeds" width="256" height="256">    |<img src="./images/seg_out.png" alt="segmented" width="256" height="256"> |




# semi-supervised-segmentation-on-graphs
- This is a reporduction of [this work](https://hal.archives-ouvertes.fr/hal-00365431):

- It solves time-dependent eikonal equation using GPU backend (pyopencl).
- <img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;f}{\partial&space;t}&space;=&space;\mathbf{1}&space;-&space;\|\nabla_{w}^{-}f\|_{\infty}" title="\frac{\partial f}{\partial t} = \mathbf{1} - \|\nabla_{w}^{-}f\|_{\infty}" />
- [Render the notebook here](https://nbviewer.jupyter.org/github/aGIToz/semi-supervised-segmentation-on-graphs/blob/main/eikonal_graph.ipynb?flush_cache=true).

# Installation.
- One needs mainly `pyopencl` and `bufferkdtree` library to create the knn-graph and run the pde on GPU.
- `pip install -r requirements.txt` should work.

# Cite this:
```latex
@InProceedings{10.1007/978-3-642-02256-2_16,
author="Ta, Vinh-Thong
and Elmoataz, Abderrahim
and L{\'e}zoray, Olivier",
title="Adaptation of Eikonal Equation over Weighted Graph",
booktitle="Scale Space and Variational Methods in Computer Vision",
year="2009",
publisher="Springer Berlin Heidelberg",
pages="187--199"
}
```
