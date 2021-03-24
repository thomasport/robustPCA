import numpy as np
from skimage.transform import resize
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import pyplot as plt

def col_to_img(mat_X, width = 168, height = 192):

    n_row, n_col = mat_X.shape
    imgs = np.zeros((n_col,height,width))
    for idx in range(n_col):
        imgs[idx,:,:] = mat_X[:,idx].reshape(width,height).T
    
    return imgs

def resampling_img(mat_X, factor, old_width = 168,old_height = 192):
  new_width = old_width/factor
  new_height = old_height/factor
  
  n_row, n_col = mat_X.shape
  new_X = np.zeros((new_width*new_height,n_col))
  
  for idx,sample in enumerate(mat_X.T):
    new_X[:,idx] = resize(mat_X[:,idx].reshape(old_width,old_height).T,(new_height,new_width)).T.flatten()
  
  return new_X

def sample_imgs(data,faces_per_individual,n_faces):
  count_faces = 0
  selected_faces = np.zeros((data.shape[0],faces_per_individual*len(n_faces)))
  # print(data[:,count_faces:count_faces+n_faces[0]].shape)
  for idx, n in enumerate(n_faces):
    selected_faces[:,idx*faces_per_individual:idx*faces_per_individual+faces_per_individual] = data[:,count_faces:count_faces+faces_per_individual]
    count_faces+=n

  return selected_faces

def plot_LS(L, S ,images, n_faces = 3):
  L_imgs = col_to_img(L)
  S_imgs = col_to_img(S)
  imgs = col_to_img(images)

  columns = 3
  rows = n_faces

  ims = []
  for i in range(n_faces):
    ims.append(imgs[i])
    ims.append(L_imgs[i])
    ims.append(S_imgs[i])
  

  fig = plt.figure(figsize=(9, 13))
  grid = ImageGrid(fig, 111,  
                 nrows_ncols=(n_faces, 3),  
                 axes_pad=0.1,  
                 )
  im = []
  for ax,im in zip(grid,ims):
      img = ax.imshow(im)
      img.set_cmap('gray')
      plt.axis('off')
  plt.show()