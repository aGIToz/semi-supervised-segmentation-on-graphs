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
@ARTICLE{5676212,
  author={Ta, Vinh-Thong and Elmoataz, Abderrahim and Lezoray, Olivier},
  journal={IEEE Transactions on Image Processing}, 
  title={Nonlocal PDEs-Based Morphology on Weighted Graphs for Image and Data Processing}, 
  year={2011},
  volume={20},
  number={6},
  pages={1504-1516},
  doi={10.1109/TIP.2010.2101610}}
```
