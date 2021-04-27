| Image     | Seeds| Segmented |
| ----------- | ----------- | ----------- |
|<img src="./data/chien_color_crop2.png" alt="org_img" width="256" height="256">   | <img src="./data/out_chien.png" alt="seeds" width="256" height="256">    |<img src="./data/seg_out.png" alt="segmented" width="256" height="256"> |




# semi-supervised-segmentation-on-graphs
- This is a reporduction of this work:

- It solves time-dependent eikonal equation using GPU backend (pyopencl).

- Render the jupyternotebook here.

# Installation.
- One needs mainly `pyopencl` and `bufferkdtree` library to create the knn-graph and run the pde on GPU.
- `pip install -r requirements.txt` should work.

# Cite this:
