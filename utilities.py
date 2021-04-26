"""
Basic utilities to preprocess and postprocess.
"""
import numpy as np
import skimage
from matplotlib import pyplot as plt
import open3d as o3d

def roll(array,k,n):
    """
    TODO: azad mar. 04 févr. 2020 14:07:39 CET
    add a desciption
    """
    mat = np.ones([n,k])
    for  i in range(n): mat[i] = array[i*k:(i*k)+k]
    return mat

def unroll(data_mat):
    """
    TODO: azad mar. 04 févr. 2020 14:08:20 CET
    add a description
    """
    n, d = data_mat.shape
    data_vec = np.ndarray(shape=(n*d,), dtype=np.float32)
    for i in range(n): data_vec[i*d: (i+1)*d] = data_mat[i]
    return data_vec

def toFmat(img):
    """
    TODO: azad lun. 03 févr. 2020 16:05:29 CET
    Converts to your feature matrix (N by c)
    pde-graph only accepts the variables in the numpy matrix format.
    """
    try:
        l, w, c = img.shape
        fmat = np.ones((l*w,c)) 
        for i in range(c): fmat[:,i] = np.reshape(img[:,:,i],(l*w,))
    except ValueError:
        try:
            l, w = img.shape
            fmat = np.ones((l*w,1)) 
            fmat = np.reshape(img,(l*w,1))
        except ValueError:
            print("Image should be at least a 2D array.")
    return fmat

def pMat(shape):
    """
    TODO: azad mer. 05 févr. 2020 10:20:41 CET
    Returns the position feature matrix.
    """
    x = np.arange(0,shape[1],1)
    y = np.arange(shape[0],0,-1)
    meshx, meshy = np.meshgrid(x,y)
    x = np.reshape(meshx,(shape[0]*shape[1],1)) 
    y = np.reshape(meshy,(shape[0]*shape[1],1)) 
    pmat = np.concatenate((x,y),axis=1) / max(shape[0],shape[1])
    return pmat

def get_wgStats(wg):
    """
    TODO: azad mar. 04 févr. 2020 14:09:47 CET
    Adds some white guassian noise
    """
    """
    >>>>NOTE<<<<: azad mar. 04 févr. 2020 14:37:06 CET
    I didn't test the code for the non-square images
    One is likely to mess up while loading or while  
    displaying the images.
    Use the tif format.
    """
    return f"The max is {np.max(wg)}; The min is {np.min(wg)}; The std is {np.std(wg)};  The median is {np.median(wg)}; The mean is {np.mean(wg)}"

def add_noise(mu, img):
    """
    TODO: azad mar. 04 févr. 2020 14:09:47 CET
    Adds some white guassian noise
    """
    """
    >>>>NOTE<<<<: azad mar. 04 févr. 2020 14:37:06 CET
    I didn't test the code for the non-square images
    One is likely to mess up while loading or while  
    displaying the images.
    Use the tif format.
    """
    mu = (mu * np.max(img))/100 # TAKE VALUE ACC TO THE MAX VALUE IN IMAGE
    noise = np.random.normal(0, mu, img.shape)
    signal = noise + img
    return signal

def toImg(fmat, shape):
    """
    TODO: azad mar. 04 févr. 2020 14:10:52 CET
    Converts from fmat to img.
    """
    img = np.zeros(shape)
    try:
        l, w, c = img.shape
        for i in range(c): img[:,:,i] = np.reshape(fmat[0:,i],(l,w))
    except ValueError:
        try:
            l, w = img.shape
            img = np.reshape(fmat,(l,w))
        except ValueError:
            print("Image should be at least a 2D array.")
    return img

def getPSNR(imgr, imgt):    
    """
    TODO: azad mar. 04 févr. 2020 14:23:45 CET
    This only works if the images are in 0,255
    Make it work for any range?
    """
    mse = np.mean( (imgr - imgt) ** 2 )
    if mse == 0: return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def imgPatches(img, patch_shape=(3, 3), **kwargs):
    """
    TODO: azad mar. 04 févr. 2020 14:11:19 CET
    Generates the image patches.
    """
    try:
        h, w, d = img.shape
    except ValueError:
        try:
            h, w = img.shape
            d = 0
        except ValueError:
            print("Image should be at least a 2D array.")
    try:
        r, c = patch_shape
    except ValueError:
        r = patch_shape[0]
        c = r
    pad_width = [(int((r - 0.5) / 2.), int((r + 0.5) / 2.)),
                 (int((c - 0.5) / 2.), int((c + 0.5) / 2.))]
    if d == 0:
        window_shape = (r, c)
        d = 1  # For the reshape in the return call
    else:
        pad_width += [(0, 0)]
        window_shape = (r, c, d)

    # Pad the image.
    img = np.pad(img, pad_width=pad_width, mode='symmetric')

    # Extract patches as node features.
    patches = skimage.util.view_as_windows(img, window_shape=window_shape)
    patches = patches.reshape((h * w, r * c * d))
    return patches

def dispImg(img): 
    """
    This does the min-max scaling the images to 0 and 1
    """
    try:
        h, w, d = img.shape
        img_tmp = (img - np.min(img)) / (np.max(img) - np.min(img))
        plt.matshow(img_tmp)
        plt.show(block=False)
    except ValueError:
        try:
            h, w = img.shape
            img_tmp = (img - np.min(img)) / (np.max(img) - np.min(img))
            plt.matshow(img_tmp, cmap = "gray")
            plt.show(block=False)
        except ValueError:
            print("Image should be at least a 2D array.")
    return None  

def downsamplepcd(**kwargs):
    """
    TODO: azad mar. 04 févr. 2020 14:11:04 CET
    display the pointclouds in open3d
    """
    p = kwargs["position"]
    t = kwargs["texture"]
    voxel_size = kwargs["voxel_size"]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p)
    pcd.colors = o3d.utility.Vector3dVector(t)
    downpcd = pcd.voxel_down_sample(voxel_size)
    p = downpcd.points
    t = downpcd.colors
    return np.asarray(p), np.asarray(t)

def returnsPCD(**kwargs):
    """
    Saves from numpy to ply file
    parmas position: The xyz coords
    parmas texture: The singal on the xyz coords
    return A pcd object in open3d
    """
    p = kwargs["position"]
    t = kwargs["texture"]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p)
    pcd.colors = o3d.utility.Vector3dVector(t)
    return pcd

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    return inlier_cloud, outlier_cloud
                                  
def writePCD(**kwargs):
    """
    Saves from numpy to ply file
    parmas position: The xyz coords
    parmas texture: The singal on the xyz coords
    parmas name: The name of the file, needs to be a string
    return A ply file shall be saved
    """
    p = kwargs["position"]
    t = kwargs["texture"]
    name = kwargs["name"]
    assert isinstance(name, str)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p)
    pcd.colors = o3d.utility.Vector3dVector(t)
    o3d.io.write_point_cloud(f"{name}.ply",pcd)
    print(f"The pcd is saved")
    return None

def displayJSur(**kwargs):
    """
    TODO: azad mar. 04 févr. 2020 14:11:04 CET
    display the pointclouds in open3d
    """
    p = kwargs["position"]
    t = kwargs["texture"]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p)
    pcd.colors = o3d.utility.Vector3dVector(t)
    visualizer = o3d.JVisualizer()
    visualizer.add_geometry(pcd)
    visualizer.show()
    return None

def displaySur(**kwargs):
    """
    TODO: azad mar. 04 févr. 2020 14:11:04 CET
    display the pointclouds in open3d
    """
    p = kwargs["position"]
    t = kwargs["texture"]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p)
    pcd.colors = o3d.utility.Vector3dVector(t)
    o3d.visualization.draw_geometries([pcd]) 
    return None

def getPositionTexture(file_path):
    """
    TODO: azad mar. 04 févr. 2020 14:11:04 CET
    display the pointclouds in open3d
    """
    mesh = o3d.io.read_triangle_mesh(file_path)
    vertices = np.asarray(mesh.vertices)
    texture = np.asarray(mesh.vertex_colors) if len(np.asarray(mesh.vertex_colors)) != 0 else np.ones(vertices.shape)*0.5
    triangles = np.asarray(mesh.triangles)
    return (vertices, texture, triangles)


def dispMesh(**kwargs):
    p = kwargs["position"]
    t = kwargs["texture"]
    f = kwargs["faces"]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(p)
    mesh.vertex_colors = o3d.utility.Vector3dVector(t)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    o3d.visualization.draw_geometries([mesh])
    return None




def genInitialDistances(**kwargs):
    """TODO: Docstring for genInitialSeeds.
    For segmentation
    :arg1: TODO
    :returns: TODO
    """
    """
    Front = [{pixel:img}]
    stack = []
    for x in 
        for y in 
            if not any(img[x,y])
            else:
                if value is in stack
                    get the image correspon to this value (from the key value)
                    and edit this image to have the value at the x,y
                else:
                    stack.append (with a key value)
                    creat a new-image
                    add that pixel to this new_img

    easy:
        just get the stack of the diff values
        use img[img==[stack]] = 1
    """
    M = []
    for i in range(np.max(labels)+1):
        mask = (labels == i)
        dist = np.ones(len(mask))
        M.append((mask,dist))

    Front = []
    for mTuple in M:
        j = 0
        for i in range(len(mTuple[0])):
            if j < num_seeds:
                if mTuple[0][i] == True:
                    mTuple[1][i] = -1
                    j = j+1
            else: break
        Front.append(-1 * np.reshape(mTuple[1], (len(mTuple[1]),1)))
    del M # not required
    print("Success!")
    return Front

def getDist2(img):
    """
    TODO: azad mar. 18 févr. 2020 11:10:29 CET
    Temporary function must get the initial distances
    for all seeds in once.
    """
    mask = (img[:,:,:] != 0).astype("int")
    mask = np.sum(mask, axis=2)
    mask[mask[:,:] == 0] = -1
    mask[mask[:,:] > 0] = 1
    return mask

def getDistPcd(texture):
    """
    TODO: azad mar. 18 févr. 2020 11:10:29 CET
    Temporary function must get the initial distances
    for all seeds in once.
    """
    mask = (texture[:,:] != 0).astype("int")
    mask = np.sum(mask, axis=1)
    mask[mask[:] == 0] = -1
    mask[mask[:] > 0] = 0
    mask[mask[:] == -1] = 1
    return mask

def getDist(img):
    """
    TODO: azad mar. 18 févr. 2020 11:10:29 CET
    Temporary function must get the initial distances
    for all seeds in once.
    """
    mask = (img[:,:,:] != 0).astype("int")
    mask = np.sum(mask, axis=2)
    mask[mask[:,:] == 0] = -1
    mask[mask[:,:] > 0] = 0
    mask[mask[:,:] == -1] = 1
    return mask

def genInitialSeeds(**kwargs):
    """TODO: Docstring for genInitialSeeds.
    For classification
    :arg1: TODO
    :returns: TODO
    """
    labels = kwargs["labels"]
    num_seeds = kwargs["num_seeds"]

    M = []
    for i in range(np.max(labels)+1):
        mask = (labels == i)
        dist = np.ones(len(mask))
        M.append((mask,dist))

    Front = []
    for mTuple in M:
        j = 0
        for i in range(len(mTuple[0])):
            if j < num_seeds:
                if mTuple[0][i] == True:
                    mTuple[1][i] = -1
                    j = j+1
            else: break
        Front.append(-1 * np.reshape(mTuple[1], (len(mTuple[1]),1)))
    del M # not required
    print("Success!")
    return Front
    
########################################################

"""
>>>>NOTE<<<<: azad mer. 05 févr. 2020 09:11:46 CET
The display of surfaces and images can be done directly
in the example files. So is the surface generation.
"""

def generateSurfave():
    """
    TODO: azad mar. 04 févr. 2020 14:12:26 CET
    Generates a surface from images.
    """
    pass

def imgtoPointCloud():
    """
    TODO: azad mar. 04 févr. 2020 14:43:38 CET
    This is useful at times.
    """
