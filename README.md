# Image 3D - Haralick in an amorphous region

## Quantization

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## Texture analysis Haralick 3D

```python
import foobar

from IPython.display import display, HTML

subject_features = pd.DataFrame()
subject_features_space = pd.DataFrame()
subject_features_space2 = pd.DataFrame()
subject_features_space3 = pd.DataFrame()
subject_features_histogram = pd.DataFrame()
subject_features_histogram2 = pd.DataFrame()
subject_features_histogram0 = pd.DataFrame()
subject_features_haralick_d1 = pd.DataFrame()

les2 = pd.DataFrame()
les42 = pd.DataFrame()
les62 = pd.DataFrame()
les5 = pd.DataFrame()
les8 = pd.DataFrame()
har_roi = np.zeros((26, 13))
har_box = np.zeros((26, 13))
lesion_3d_features30 = pd.DataFrame()

# Declara la lesion
# Lesion1 = np.random.randint(8, size=(7,8,8))

# ROI Region of interest 3d

for roi_i in range(1, n_labels + 1):

  roi_indexes = np.where(labeled_dm == roi_i)
  roi_indexes1 = np.logical_or(labeled_dm == roi_i, labeled_dm == roi_i)
  roi_indexes1 = nb.Nifti1Image(roi_indexes1.astype(np.uint64), flair_image.affine)
  roi_indexes1_array = roi_indexes1.get_fdata()
  roi_indexes1_array =  roi_indexes1_array.astype(np.uint64)


  x_min, x_max = roi_indexes[0].min(), roi_indexes[0].max()
  y_min, y_max = roi_indexes[1].min(), roi_indexes[1].max()
  z_min, z_max = roi_indexes[2].min(), roi_indexes[2].max()

  # crea el cubo con la roi y los valores fuera de la roi cero.
  mic = roi_indexes1_array[x_min:(x_max+1), y_min:(y_max+1), z_min:(z_max+1)] * flair_array[x_min:(x_max+1), y_min:(y_max+1), z_min:(z_max+1) ]
  mic100 = roi_indexes1_array[x_min:(x_max+1), y_min:(y_max+1), z_min:(z_max+1)] 
  mic2 = ndimage.binary_erosion(mic100).astype(mic100.dtype)
  mic3 =  mic100 - mic2

  

  mic300=   mic3 * flair_array[x_min:(x_max+1), y_min:(y_max+1), z_min:(z_max+1) ]
  print(mic300.shape)
  #print(mic.shape)
  #print(mic)
  #print(mic100)
  # Recorre el cubo y a los valores fuera de la roi le da el valor -1

  #for x in range(x_max-x_min):
  #  for y in range(y_max-y_min):
  #    for z in range(z_max-z_min):
  #      if Lesion1[x][y][z] = 0:        


  mic10 = mic[np.where(mic > 0)]

  if mic10.sum == 0:
    mic1=0
  else:
    mic1 = pd.Series(mic10.ravel())


  mic30 = mic300[np.where(mic300 > 0)]

  if mic30.sum == 0:
    mic4=0
  else:
    mic4 = pd.Series(mic30.ravel())

  lesion =  mic

  # Features spatial

  centroid_axis0, centroid_axis1, centroid_axis2  = ndimage.center_of_mass(labeled_dm == roi_i)

  centroid_axis02, centroid_axis12, centroid_axis22  = ndimage.center_of_mass(mic100)

  centroid_axis03, centroid_axis13, centroid_axis23  = ndimage.center_of_mass(mic)

  space1 = pd.DataFrame()
  space = pd.Series()
  space['centroid_axis0'] = centroid_axis0
  space['centroid_axis1'] = centroid_axis1
  space['centroid_axis2'] = centroid_axis2
  space['axis0_min'] = x_min
  space['axis0_max'] = x_max
  space['axis1_min'] = y_min
  space['axis1_max'] = y_max
  space['axis2_min'] = z_min
  space['axis2_max'] = z_max
  space.name = f'Lesion_{roi_i}'
  
  print('-------------------------------')
  print(f'Lesion: {roi_i}')
  print('Features extraction')
  space1 =  space1.append(space, ignore_index=False)
  subject_features_space =  subject_features_space.append(space, ignore_index=False)
  print('1. Spaces features: Position')


  print(display(HTML(space1.to_html())))

  #li = mic100*mic100[centroid_axis03,centroid_axis13,centroid_axis23]
  plt.figure()
  plt.xticks(np.arange(-.5, (y_max-y_min), 1))
  plt.yticks(np.arange(-.5, (x_max-x_min), 1))
  plt.imshow(mic300[:,:, int(centroid_axis23)], cmap='gray')
  plt.grid(True)
  plt.show()


  plt.figure()
  plt.xticks(np.arange(-.5, (y_max-y_min), 1))
  plt.yticks(np.arange(-.5, (x_max-x_min), 1))
  plt.imshow(mic[:,:, int(centroid_axis23)], cmap='gray')
  plt.scatter(centroid_axis13,centroid_axis03,color='r')
  plt.scatter(centroid_axis12,centroid_axis02,color='b')
  plt.grid(True)
  plt.show()



  space12 = pd.DataFrame()
  space2 = pd.Series()
  space2['centroid_axis0'] = centroid_axis02
  space2['centroid_axis1'] = centroid_axis12
  space2['centroid_axis2'] = centroid_axis22
  space2.name = f'Lesion_{roi_i}'
  
  print('-------------------------------')
  print(f'Lesion: {roi_i}')
  print('Features extraction')
  space12 =  space12.append(space2, ignore_index=False)
  subject_features_space2 =  subject_features_space2.append(space2, ignore_index=False)
  print('1. Spaces features: Genometric center')
  print(mic[int(centroid_axis02),int(centroid_axis12),int(centroid_axis22)])

  print(display(HTML(space12.to_html())))


  space13 = pd.DataFrame()
  space3 = pd.Series()
  space3['centroid_axis0'] = centroid_axis03
  space3['centroid_axis1'] = centroid_axis13
  space3['centroid_axis2'] = centroid_axis23
  space3.name = f'Lesion_{roi_i}'
  
  print('-------------------------------')
  print(f'Lesion: {roi_i}')
  print('Features extraction')
  space13 =  space13.append(space3, ignore_index=False)
  subject_features_space3 =  subject_features_space3.append(space3, ignore_index=False)
  print('1. Spaces features: Center of mass')
  centerint= mic[int(centroid_axis03),int(centroid_axis13),int(centroid_axis23)]
  print(f'Center of Mass = {centerint}')


  print(display(HTML(space13.to_html())))

  # Histogram Lesion 
  if mic1.empty:
    nobs1
    min_value1 = 0
    max_value1 = 0
    mean1 = 0
    var1 = 0
    skew1 = 0
    kurt1 = 0
  else:
    nobs1, (min_value1, max_value1), mean1, var1, skew1, kurt1 = stats.describe(mic1, nan_policy='propagate')
    print(f'min_lesion: {min_value1}')

  dm_features1 = pd.DataFrame()
  dm_features = pd.Series()
  
  dm_features['Volume'] = nobs1  # No. de voxeles
  dm_features['Minimum'] = round(min_value1, 2)
  dm_features['Maximum'] = round(max_value1, 2)
  dm_features['Mean'] = round(mean1, 2)
  dm_features['Variance'] = round(var1, 2)
  dm_features['Skewness'] = round(skew1, 2)
  dm_features['Kurtosis'] = round(kurt1, 2)
  dm_features.name = f'Lesion_{roi_i}'
  
  dm_features1 = dm_features1.append(dm_features, ignore_index=False)
  subject_features_histogram =  subject_features_histogram.append(dm_features, ignore_index=False)

  
  print('2. Histogram Features')
  
  print(display(HTML(dm_features1.to_html())))










  # Histogram Lesion 
  if mic4.empty:
    nobs10
    min_value10 = 0
    max_value10 = 0
    mean10 = 0
    var10 = 0
    skew10 = 0
    kurt10 = 0
  else:
    nobs10, (min_value10, max_value10), mean10, var10, skew10, kurt10 = stats.describe(mic4, nan_policy='propagate')
    print(f'min_lesion: {min_value10}')

  dm_features10 = pd.DataFrame()
  dm_features0 = pd.Series()
  
  dm_features0['Volume'] = nobs10  # No. de voxeles
  dm_features0['Minimum'] = round(min_value10, 2)
  dm_features0['Maximum'] = round(max_value10, 2)
  dm_features0['Mean'] = round(mean10, 2)
  dm_features0['Variance'] = round(var10, 2)
  dm_features0['Skewness'] = round(skew10, 2)
  dm_features0['Kurtosis'] = round(kurt10, 2)
  dm_features0.name = f'Lesion_{roi_i}'
  
  dm_features10 = dm_features10.append(dm_features0, ignore_index=False)
  subject_features_histogram0 =  subject_features_histogram0.append(dm_features0, ignore_index=False)

  
  print('2. Histogram Features Border')
  
  print(display(HTML(dm_features10.to_html())))

  descriptor = centerint - mean10

  print(f'Descriptor= {descriptor}')












 # Cuantizacion 8 por lesion 

  w100 = (max_value1 - min_value1) / 7

  face100 = lesion - min_value1

  face100[face100  <= 0] = 0

  

  Lesion1 = (face100 / w100)
  Lesion1 = Lesion1.astype(np.uint8)

  Lesion1 = Lesion1 + 1
  Lesion1 = Lesion1 * roi_indexes1_array[x_min:(x_max+1), y_min:(y_max+1), z_min:(z_max+1)]
  Lesion1 = Lesion1.astype(np.uint8)

  Lesion10 = Lesion1[np.where(Lesion1 > 0)]

  Lesion101 = pd.Series(Lesion10.ravel())

  print(f'Tamano de la lesion1{Lesion1.shape}')
  max = np.max(Lesion1)
  print(f'max1:{max}')

 # Histogram quantized

  nobs2, (min_value2, max_value2), mean2, var2, skew2, kurt2 = stats.describe(Lesion101, nan_policy='propagate')
  
  dm_features3 = pd.DataFrame()
  dm_features2 = pd.Series()
  
  dm_features2['Volume'] = nobs2  # No. de voxeles
  dm_features2['Minimum'] = round(min_value2, 2)
  dm_features2['Maximum'] = round(max_value2, 2)
  dm_features2['Mean'] = round(mean2, 2)
  dm_features2['Variance'] = round(var2, 2)
  dm_features2['Skewness'] = round(skew2, 2)
  dm_features2['Kurtosis'] = round(kurt2, 2)
  dm_features2.name = f'Lesion_{roi_i}'
  
  dm_features3 = dm_features3.append(dm_features2, ignore_index=False)
  subject_features_histogram2 =  subject_features_histogram2.append(dm_features2, ignore_index=False)


  print('2. Histogram Features quantized')
  
  print(display(HTML(dm_features3.to_html())))



  # Tamaño de la lesion
  axis0 = np.size(Lesion1, 0)
  axis1 = np.size(Lesion1, 1)
  axis2 = np.size(Lesion1, 2)
  max = np.max(Lesion1)



  ########################################
  #####Gray Level Co-Ocurrence Matrix GLCM

  # Declara la GLCM Simetrica 

  dir = 13
  # Size GLCM:
  # Gray = max gray scale
  # Gray2 = max gray scale 

  glcm = np.random.randint(9, size=(dir, max+1, max+1))
  print(f'Tamaño de GLCM: {glcm.shape}')

  i=1

  # GLCM todo los valores a cero

  for x in range(13): # GLCM
    for y in range(max+1): # Col GLCM
      for z in range(max+1): # Row GLCM
        glcm[x][y][z] = 0



  # Recorre cada voxel de la lesion y le da valor a la GLCM

  # 0. En direccion Vertical centro=x Arriba=y+1 centro=z

  for x in range(axis0): 
    for y in range(axis1): 
      for z in range(axis2): 
        if y < axis1-1:
          glcm[0][Lesion1[x][y][z]][Lesion1[x][y+1][z]] = glcm[0][Lesion1[x][y][z]][Lesion1[x][y+1][z]] + 1


  # 1. En direccion centro=x centro=y izquierda=z-1

  for x in range(axis0): 
    for y in range(axis1): 
      for z in range(axis2): 
        if z > 0:
          glcm[1][Lesion1[x][y][z]][Lesion1[x][y][z-1]] = glcm[1][Lesion1[x][y][z]][Lesion1[x][y][z-1]] + 1


  # 2. En direccion adelante=x+1 centro=y centro=z

  for x in range(axis0): 
    for y in range(axis1): 
      for z in range(axis2): 
        if x < axis0-1:
          glcm[2][Lesion1[x][y][z]][Lesion1[x+1][y][z]] = glcm[2][Lesion1[x][y][z]][Lesion1[x+1][y][z]] + 1

  # 3. En direccion Diagonal  adelante=x+1 arriba=y+1 derecha=z+1

  for x in range(axis0): 
    for y in range(axis1): 
      for z in range(axis2): 
        if x < axis0-1 and y < axis1-1 and z < axis2-1:
          glcm[3][Lesion1[x][y][z]][Lesion1[x+1][y+1][z+1]] = glcm[3][Lesion1[x][y][z]][Lesion1[x+1][y+1][z+1]] + 1


  # 4. En direccion Diagonal  adelante=x+1 arriba=y+1 izquierda=z-1

  for x in range(axis0): 
    for y in range(axis1): 
      for z in range(axis2): 
        if x < axis0-1 and y < axis1-1 and z > 0:
          glcm[4][Lesion1[x][y][z]][Lesion1[x+1][y+1][z-1]] = glcm[4][Lesion1[x][y][z]][Lesion1[x+1][y+1][z-1]] + 1


  # 5. En direccion Diagonal  adelante=x+1 abajo=y-1 derecha=z+1

  for x in range(axis0): 
    for y in range(axis1): 
      for z in range(axis2): 
        if x < axis0-1 and y > 0 and z < axis2-1:
          glcm[5][Lesion1[x][y][z]][Lesion1[x+1][y-1][z+1]] = glcm[5][Lesion1[x][y][z]][Lesion1[x+1][y-1][z+1]] + 1


  # 6. En direccion Diagonal  centro=x arriba=y+1 derecha=z+1

  for x in range(axis0): 
    for y in range(axis1): 
      for z in range(axis2): 
        if y < axis1-1  and z < axis2-1:
          glcm[6][Lesion1[x][y][z]][Lesion1[x][y+1][z+1]] = glcm[6][Lesion1[x][y][z]][Lesion1[x][y+1][z+1]] + 1


# 7. En direccion Diagonal  adelante=x+1 abajo=y-1 centro=z

  for x in range(axis0): 
    for y in range(axis1): 
      for z in range(axis2): 
        if x < axis0-1  and y > 0:
          glcm[7][Lesion1[x][y][z]][Lesion1[x+1][y-1][z]] = glcm[7][Lesion1[x][y][z]][Lesion1[x+1][y-1][z]] + 1

  # 8. En direccion Diagonal  adelante=x+1 centro=y derecha=z+1

  for x in range(axis0): 
    for y in range(axis1): 
      for z in range(axis2): 
        if x < axis0-1  and z < axis2-1:
          glcm[8][Lesion1[x][y][z]][Lesion1[x+1][y][z+1]] = glcm[8][Lesion1[x][y][z]][Lesion1[x+1][y][z+1]] + 1

  # 9. En direccion Diagonal  adelante=x+1 centro=y izquierda=z-1

  for x in range(axis0): 
    for y in range(axis1): 
      for z in range(axis2): 
        if x < axis0-1  and z > 0:
          glcm[9][Lesion1[x][y][z]][Lesion1[x+1][y][z-1]] = glcm[9][Lesion1[x][y][z]][Lesion1[x+1][y][z-1]] + 1

  # 10. En direccion Diagonal  centro=x arriba=y+1 izquierda=z-1

  for x in range(axis0): 
    for y in range(axis1): 
      for z in range(axis2): 
        if y < axis1-1 and z > 0:
          glcm[10][Lesion1[x][y][z]][Lesion1[x][y+1][z-1]] = glcm[10][Lesion1[x][y][z]][Lesion1[x][y+1][z-1]] + 1


  # 11. En direccion Diagonal  adelante=x+1 arriba=y+1 centro=z

  for x in range(axis0): 
    for y in range(axis1): 
      for z in range(axis2): 
        if x < axis0-1  and y < axis1-1:
          glcm[11][Lesion1[x][y][z]][Lesion1[x+1][y+1][z]] = glcm[11][Lesion1[x][y][z]][Lesion1[x+1][y+1][z]] + 1


  # 12. En direccion Diagonal  adelante=x+1 abajo=y-1 izquierda=z-1

  for x in range(axis0): 
    for y in range(axis1): 
      for z in range(axis2): 
        if x < axis0-1 and y > 0 and z > 0:
          glcm[12][Lesion1[x][y][z]][Lesion1[x+1][y-1][z-1]] = glcm[12][Lesion1[x][y][z]][Lesion1[x+1][y-1][z-1]] + 1


  glcm1 = np.delete(glcm,0,1)

  glcm2 = np.delete(glcm1,0,2)
  #print('GLCM Direccion 0:')

  # Matrix GLCM neighbour right

  Lesion200 = glcm2[0]


  # Matrix GLCM neighbour right Transpose

  Lesion201 = np.transpose(Lesion200)


  # Matrix symmetric

  Lesion202 = Lesion200 + Lesion201

  print(Lesion200)

  mean30 = np.max(Lesion200)/2

  plt.figure()
  plt.imshow(Lesion200, cmap='viridis')
  plt.xticks(np.arange(-.5, 7.5, 1))
  plt.yticks(np.arange(-.5, 7.5, 1))
  plt.grid(True)
  for i1 in range(max):
    for j1 in range(max):
      if Lesion200[i1,j1] > mean30 :
        plt.text(j1, i1, Lesion200[i1,j1], ha="center", va="center", color="black")
      else:
        plt.text(j1, i1, Lesion200[i1,j1], ha="center", va="center", color="w")
  plt.xlabel('Neighbour pixel value  ')
  plt.ylabel('Reference pixel value ')
  #plt.colorbar()
  plt.title('GLCM 1 Right neighbour')
  plt.show()

  
  print(Lesion201)
  mean31 = np.max(Lesion201)/2
  plt.figure()
  plt.imshow(Lesion201, cmap='viridis')
  plt.xticks(np.arange(-.5, 7.5, 1))
  plt.yticks(np.arange(-.5, 7.5, 1))
  plt.grid(True)
  for i1 in range(max):
    for j1 in range(max):
      if Lesion201[i1,j1] > mean31 :
        plt.text(j1, i1, Lesion201[i1,j1], ha="center", va="center", color="black")
      else:
        plt.text(j1, i1, Lesion201[i1,j1], ha="center", va="center", color="w")
  plt.xlabel('Reference pixel value ')
  plt.ylabel('Neighbour pixel value  ')
  #plt.colorbar()
  plt.title('GLCM 1 Right neighbour Transpose')
  plt.show()

  

  print(Lesion202)

  mean2 = np.max(Lesion202)/2

  plt.figure()
  plt.imshow(Lesion202, cmap='viridis')
  plt.xticks(np.arange(-.5, 7.5, 1))
  plt.yticks(np.arange(-.5, 7.5, 1))
  plt.grid(True)
  for i1 in range(max):
    for j1 in range(max):
      if Lesion202[i1,j1] > mean2 :
        plt.text(j1, i1, Lesion202[i1,j1], ha="center", va="center", color="black")
      else:
        plt.text(j1, i1, Lesion202[i1,j1], ha="center", va="center", color="w")
  plt.xlabel('x')
  plt.ylabel('y')
  #plt.colorbar()
  plt.title('GLCM 1 symmetric')
  plt.show()
  
  #print(np.sum(Lesion202))
  

  ########################################################
  #########HARALICK DESCRIPTOR


 # Haralicks descriptor: BOX 3D

  
  for p in range(13):
    #print(p)
    Lesion111= glcm[p] / glcm[p].sum()


    # Haralicks descriptor: BOX 3D
    #print('Haralick BOX 3D')


    #Haralick Descriptor : 1. ASM
    sum=0
    for n in range(max+1):
      for m in range(max+1):
        sum = sum + (Lesion111[n][m] * Lesion111[n][m])
    #print(f'ASM: {round(sum, 3)}')
    har_box[p][0] = sum

    #Haralick Descriptor : 2. Contrast
    sum=0
    for n in range(max+1):
      for m in range(max+1):
        sum = sum + (Lesion111[n][m] * ((n-m)*(n-m)))
    #print(f'Contrast: {round(sum, 3)}')
    har_box[p][1] = sum



    #Mean
    sum=0
    for n in range(max+1):
      nim=0
      for m in range(max+1):
        nim = nim + Lesion111[n][m]
      sum = sum + ( n * nim )
      mean1 = sum    
    #print(f'MEAN1: {round(sum, 3)}')



    #Mean
    sum=0
    for m in range(max+1):
      nim=0
      for n in range(max+1):
        nim = nim + Lesion111[n][m]
      sum = sum + ( m * nim )    
      mean2 = sum 
    #print(f'MEAN2: {round(sum, 3)}')



    #Variance 1
    sum=0
    for n in range(max+1):
      nim=0
      for m in range(max+1):
        sum = sum + ( (n - mean1)**2 * Lesion111[n][m] )
      #mean1 = sum    
    variance1 = sum
    #print(f'Variance1: {round(sum, 3)}')

    #Variance 2
    sum=0
    for m in range(max+1):
      for n in range(max+1):
        sum = sum + ( ( ( m - mean2)**2 ) * Lesion111[n][m] )    
    variance2 = sum
    #print(f'Variance2: {round(sum, 3)}')



    va= variance1 * variance2
    #Haralick Descriptor : 3. Correlation
    sum=0.0
    for m in range(max+1):
      for n in range(max+1):
        sum = sum + (Lesion111[n][m]*(  ( m - mean2)*(n - mean1)  / math.sqrt(va) ))
    #print(f'Correlation: {round(sum, 3)}')
    har_box[p][2] = sum

    #Haralick Descriptor : 4. Sum of Square
    sum=0
    #print(f'Sum of Squares:')
    har_box[p][3] = variance1 


    #Haralick Descriptor : 3. Homogenity
    sum=0
    for n in range(max+1):
      for m in range(max+1):
        sum = sum + (Lesion111[n][m] / (1+((n-m)*(n-m))))
    #print(f'Homogenity: {round(sum, 3)}')
    har_box[p][4] = sum


    #Haralick Descriptor : 6. Sum Average
    sum=0
    #print(f'Sum average:')
    har_box[p][5] = mean1 + mean2

    #Haralick Descriptor : 7. Sum Variance
    sum=0
    #print(f'Sum variance:')
    har_box[p][6] = sum

    #Haralick Descriptor : 8. Sum Entropy
    sum=0
    #print(f'Sum entropy:')
    har_box[p][7] = sum


    #Haralick Descriptor : 9. Entropy
    sum=0
    for n in range(max+1):
      for m in range(max+1):
        if Lesion111[n][m] <= 0:
          sum = sum
        else:
          sum = sum + (Lesion111[n][m] * ( (-1) * np.log(Lesion111[n][m]) ) )
    #print(f'Entropy: {round(sum, 3)}')
    har_box[p][8] = sum


    #Haralick Descriptor : 10. Difference variance
    sum=0
    #print(f'Difference variance:')
    har_box[p][9] = sum


    #Haralick Descriptor : 11. Difference entropy
    sum=0
    #print(f'Difference entropy:')
    har_box[p][10] = sum

    #Haralick Descriptor : 12. Information correlation
    sum=0
    #print(f'Information correlation:')
    har_box[p][11] = sum

    #Haralick Descriptor : 13. Coeficient correlation
    sum=0
    #print(f'Coeficient correlation:')
    har_box[p][12] = sum




  
  for p in range(13):

    # Haralicks descriptor: ROI 3D

   # print('Haralick ROI 3D')
   # print(f'GLCM:{p}')



    # GLCM fila 1 y columa 1 a 0
    # al hacer esto los valores 0 que estan dentro de la roi NO son tenidos
    # en cuenta y puede ser significativo.

    

    #glcm[p,0,:] = 0
    #glcm[p,:,0] = 0

    Lesion111= glcm2[p] / glcm2[p].sum()


    #Haralick Descriptor : 1. ASM
    sum=0
    for n in range(max):
      for m in range(max):
        sum = sum + (Lesion111[n][m] * Lesion111[n][m])    
    #print(f'ASM: {round(sum, 3)}')
    har_roi[p][0] = sum


    #Haralick Descriptor : 2. Contrast
    sum=0
    for n1 in range(max):
      for m1 in range(max):
        sum = sum + (Lesion111[n1][m1] * ((n1-m1)*(n1-m1)))
    #print(f'Contrast: {round(sum, 3)}')
    har_roi[p][1] = sum


    #Mean
    sum=0
    for m in range(max):
      nim=0
      for n in range(max):
        nim = nim + Lesion111[n][m]
      sum = sum + ( m * nim )    
      mean2 = sum 
    #print(f'MEAN2: {round(sum, 3)}')



    #Variance 1
    sum=0
    for n in range(max):
      nim=0
      for m in range(max):
        sum = sum + ( (n - mean1)**2 * Lesion111[n][m] )
      #mean1 = sum    
    variance1 = sum
    #print(f'Variance1: {round(sum, 3)}')

    #Variance 2
    sum=0
    for m in range(max):
      for n in range(max):
        sum = sum + ( ( ( m - mean2)**2 ) * Lesion111[n][m] )    
    variance2 = sum
    #print(f'Variance2: {round(sum, 3)}')



    va= variance1 * variance2
    #Haralick Descriptor : 3. Correlation
    sum=0.0
    for m in range(max):
      for n in range(max):
        sum = sum + (Lesion111[n][m]*(  ( m - mean2)*(n - mean1)  / math.sqrt(va) ))
    #print(f'Correlation: {round(sum, 3)}')
    har_roi[p][2] = sum

    #Haralick Descriptor : 4. Sum of Square
    sum=0
    #print(f'Sum of Squares:')
    har_roi[p][3] = variance1 


    #Haralick Descriptor : 3. Homogenity
    sum=0
    for n in range(max):
      for m in range(max):
        sum = sum + (Lesion111[n][m] / (1+((n-m)*(n-m))))
    #print(f'Homogenity: {round(sum, 3)}')
    har_roi[p][4] = sum


    #Haralick Descriptor : 6. Sum Average
    sum=0
    #print(f'Sum average:')
    har_roi[p][5] = mean1 + mean2

    #Haralick Descriptor : 7. Sum Variance
    sum=0
    #print(f'Sum variance:')
    har_roi[p][6] = sum

    #Haralick Descriptor : 8. Sum Entropy
    sum=0
    #print(f'Sum entropy:')
    har_roi[p][7] = sum


    #Haralick Descriptor : 9. Entropy
    sum=0
    for n in range(max):
      for m in range(max):
        if Lesion111[n][m] <= 0:
          sum = sum
        else:
          sum = sum + (Lesion111[n][m] * ( (-1) * np.log(Lesion111[n][m]) ) )
    #print(f'Entropy: {round(sum, 3)}')
    har_roi[p][8] = sum



    #Haralick Descriptor : 10. Difference variance
    sum=0
    #print(f'Difference variance:')
    har_roi[p][9] = sum



    #Haralick Descriptor : 11. Difference entropy
    sum=0
    #print(f'Difference entropy:')
    har_roi[p][10] = sum


    #Haralick Descriptor : 12. Information correlation
    sum=0
    #print(f'Information correlation:')
    har_roi[p][11] = sum

    #Haralick Descriptor : 13. Coeficient correlation
    sum=0
    #print(f'Coeficient correlation:')
    har_roi[p][12] = sum
    
    #if p==0:
      #print(har_roi[p])



  #print('Haralick Mahotas')
  if x_max-x_min==0 or y_max-y_min==0 or z_max-z_min==0:
    h_feature = 0
  else:
    print(Lesion1.shape)
    h_feature = mt.features.haralick(Lesion1) 
  #print(h_feature)
  #print(f'ASM: {round(h_feature[0][0], 3)}')
  #print(f'Contrast: {round(h_feature[0][1], 3)}')
  #print(f'Correlation: {round(h_feature[0][2], 3)}')
  #print(f'Sum of Squares: {round(h_feature[0][3], 3)}')
  #print(f'Homogenity: {round(h_feature[0][4], 3)}')
  #print(f'Sum average: {round(h_feature[0][5], 3)}')
  #print(f'Sum variance: {round(h_feature[0][6], 3)}')
  #print(f'Sum entropy: {round(h_feature[0][7], 3)}')
  #print(f'Entropy: {round(h_feature[0][8], 3)}')
  #print(f'Difference variance: {round(h_feature[0][9], 3)}')
  #print(f'Difference entropy: {round(h_feature[0][10], 3)}')
  #print(f'Information correlation: {round(h_feature[0][11], 3)}')
  #print(f'Coeficient correlation: {round(h_feature[0][12], 3)}')


### HARALICK MAHOTAS

  lesion_3d_features = pd.DataFrame()
  
  haralick_lesion_3d = ['1 Angular second moment', '2 Contrast', '3 Correlation', '4 Sum of squares', '5 Inverse difference moment', '6 Sum average', '7 Sum variance', '8 Sum entropy', '9 Entropy', '10 Difference variance', '11 Diference entropy', '12 Information correlation','13 Coefficient correlation']
  haralick_lesion_3d0= ['1-1 Angular second moment', '1-2 Contrast', '1-3 Correlation', '1-4 Sum of squares', '1-5 Inverse difference moment', '1-6 Sum average', '1-7 Sum variance', '1-8 Sum entropy', '1-9 Entropy', '1-10 Difference variance', '1-11 Diference entropy', '1-12 Information correlation','1-13 Coefficient correlation',
                         '2-1 Angular second moment', '2-2 Contrast', '2-3 Correlation', '2-4 Sum of squares', '2-5 Inverse difference moment', '2-6 Sum average', '2-7 Sum variance', '2-8 Sum entropy', '2-9 Entropy', '2-10 Difference variance', '2-11 Diference entropy', '2-12 Information correlation','2-13 Coefficient correlation', 
                         '3-1 Angular second moment', '3-2 Contrast', '3-3 Correlation', '3-4 Sum of squares', '3-5 Inverse difference moment', '3-6 Sum average', '3-7 Sum variance', '3-8 Sum entropy', '3-9 Entropy', '3-10 Difference variance', '3-11 Diference entropy', '3-12 Information correlation','3-13 Coefficient correlation',
                         '4-1 Angular second moment', '4-2 Contrast', '4-3 Correlation', '4-4 Sum of squares', '4-5 Inverse difference moment', '4-6 Sum average', '4-7 Sum variance', '4-8 Sum entropy', '4-9 Entropy', '4-10 Difference variance', '4-11 Diference entropy', '4-12 Information correlation','4-13 Coefficient correlation',
                         '5-1 Angular second moment', '5-2 Contrast', '5-3 Correlation', '5-4 Sum of squares', '5-5 Inverse difference moment', '5-6 Sum average', '5-7 Sum variance', '5-8 Sum entropy', '5-9 Entropy', '5-10 Difference variance', '5-11 Diference entropy', '5-12 Information correlation','5-13 Coefficient correlation', 
                         '6-1 Angular second moment', '6-2 Contrast', '6-3 Correlation', '6-4 Sum of squares', '6-5 Inverse difference moment', '6-6 Sum average', '6-7 Sum variance', '6-8 Sum entropy', '6-9 Entropy', '6-10 Difference variance', '6-11 Diference entropy', '6-12 Information correlation','6-13 Coefficient correlation',
                         '7-1 Angular second moment', '7-2 Contrast', '7-3 Correlation', '7-4 Sum of squares', '7-5 Inverse difference moment', '7-6 Sum average', '7-7 Sum variance', '7-8 Sum entropy', '7-9 Entropy', '7-10 Difference variance', '7-11 Diference entropy', '7-12 Information correlation','7-13 Coefficient correlation',
                         '8-1 Angular second moment', '8-2 Contrast', '8-3 Correlation', '8-4 Sum of squares', '8-5 Inverse difference moment', '8-6 Sum average', '8-7 Sum variance', '8-8 Sum entropy', '8-9 Entropy', '8-10 Difference variance', '8-11 Diference entropy', '8-12 Information correlation','8-13 Coefficient correlation', 
                         '9-1 Angular second moment', '9-2 Contrast', '9-3 Correlation', '9-4 Sum of squares', '9-5 Inverse difference moment', '9-6 Sum average', '9-7 Sum variance', '9-8 Sum entropy', '9-9 Entropy', '9-10 Difference variance', '9-11 Diference entropy', '9-12 Information correlation','9-13 Coefficient correlation',
                         '10-1 Angular second moment', '10-2 Contrast', '10-3 Correlation', '10-4 Sum of squares', '10-5 Inverse difference moment', '10-6 Sum average', '10-7 Sum variance', '10-8 Sum entropy', '10-9 Entropy', '10-10 Difference variance', '10-11 Diference entropy', '10-12 Information correlation','10-13 Coefficient correlation',
                         '11-1 Angular second moment', '11-2 Contrast', '11-3 Correlation', '11-4 Sum of squares', '11-5 Inverse difference moment', '11-6 Sum average', '11-7 Sum variance', '11-8 Sum entropy', '11-9 Entropy', '11-10 Difference variance', '11-11 Diference entropy', '11-12 Information correlation','11-13 Coefficient correlation', 
                         '12-1 Angular second moment', '12-2 Contrast', '12-3 Correlation', '12-4 Sum of squares', '12-5 Inverse difference moment', '12-6 Sum average', '12-7 Sum variance', '12-8 Sum entropy', '12-9 Entropy', '12-10 Difference variance', '12-11 Diference entropy', '12-12 Information correlation','12-13 Coefficient correlation',
                         '13-1 Angular second moment', '13-2 Contrast', '13-3 Correlation', '13-4 Sum of squares', '13-5 Inverse difference moment', '13-6 Sum average', '13-7 Sum variance', '13-8 Sum entropy', '13-9 Entropy', '13-10 Difference variance', '13-11 Diference entropy', '13-12 Information correlation','13-13 Coefficient correlation']

  
  if(axis0 > 1 and axis1 > 1 and axis2 > 1):
    les = pd.DataFrame()  
    les1 = pd.Series()
    
    for v in range(0, 13):     
      for w in range(0, 13):
        les1 = np.concatenate((les1, h_feature[v][w]), axis=None)
        les1=pd.Series(les1)
           
      haralick3d_dict2 =  {f'h_{haralick_lesion_3d[i]}': val for i, val in enumerate(h_feature[v])}    
      haralick_lesion_3d_series = pd.Series({**haralick3d_dict2}, name=f'Lesion_{roi_i}_{v}')
      lesion_3d_features = lesion_3d_features.append(haralick_lesion_3d_series)  
      subject_features_haralick_d1 =  subject_features_haralick_d1.append(haralick_lesion_3d_series, ignore_index=False)

    les = les.append(les1, ignore_index=True)
    les2 = les2.append(les)
    haralick3d_dict = {f'h_{haralick_lesion_3d[i]}': val for i, val in enumerate(h_feature.mean(axis=0))}
  
  else:
    h_feature = {}
    haralick3d_dict = {f'h_{haralick_lesion_3d[i]}': 0 for i, val in enumerate(haralick_lesion_3d)}

  print('1. MAHOTAS library Haralick ')
  print(display(HTML(les.to_html())))
  print(display(HTML(lesion_3d_features.to_html())))



### HARALICK BOX

  lesion_3d_features = pd.DataFrame()
  haralick_lesion_3d = ['1 Angular second moment', '2 Contrast', '3 Correlation', '4 Sum of squares', '5 Inverse difference moment', '6 Sum average', '7 Sum variance', '8 Sum entropy', '9 Entropy', '10 Difference variance', '11 Diference entropy', '12 Information correlation','13 Coefficient correlation']
  haralick_lesion_3d0= ['1-1 Angular second moment', '1-2 Contrast', '1-3 Correlation', '1-4 Sum of squares', '1-5 Inverse difference moment', '1-6 Sum average', '1-7 Sum variance', '1-8 Sum entropy', '1-9 Entropy', '1-10 Difference variance', '1-11 Diference entropy', '1-12 Information correlation','1-13 Coefficient correlation',
                         '2-1 Angular second moment', '2-2 Contrast', '2-3 Correlation', '2-4 Sum of squares', '2-5 Inverse difference moment', '2-6 Sum average', '2-7 Sum variance', '2-8 Sum entropy', '2-9 Entropy', '2-10 Difference variance', '2-11 Diference entropy', '2-12 Information correlation','2-13 Coefficient correlation', 
                         '3-1 Angular second moment', '3-2 Contrast', '3-3 Correlation', '3-4 Sum of squares', '3-5 Inverse difference moment', '3-6 Sum average', '3-7 Sum variance', '3-8 Sum entropy', '3-9 Entropy', '3-10 Difference variance', '3-11 Diference entropy', '3-12 Information correlation','3-13 Coefficient correlation',
                         '4-1 Angular second moment', '4-2 Contrast', '4-3 Correlation', '4-4 Sum of squares', '4-5 Inverse difference moment', '4-6 Sum average', '4-7 Sum variance', '4-8 Sum entropy', '4-9 Entropy', '4-10 Difference variance', '4-11 Diference entropy', '4-12 Information correlation','4-13 Coefficient correlation',
                         '5-1 Angular second moment', '5-2 Contrast', '5-3 Correlation', '5-4 Sum of squares', '5-5 Inverse difference moment', '5-6 Sum average', '5-7 Sum variance', '5-8 Sum entropy', '5-9 Entropy', '5-10 Difference variance', '5-11 Diference entropy', '5-12 Information correlation','5-13 Coefficient correlation', 
                         '6-1 Angular second moment', '6-2 Contrast', '6-3 Correlation', '6-4 Sum of squares', '6-5 Inverse difference moment', '6-6 Sum average', '6-7 Sum variance', '6-8 Sum entropy', '6-9 Entropy', '6-10 Difference variance', '6-11 Diference entropy', '6-12 Information correlation','6-13 Coefficient correlation',
                         '7-1 Angular second moment', '7-2 Contrast', '7-3 Correlation', '7-4 Sum of squares', '7-5 Inverse difference moment', '7-6 Sum average', '7-7 Sum variance', '7-8 Sum entropy', '7-9 Entropy', '7-10 Difference variance', '7-11 Diference entropy', '7-12 Information correlation','7-13 Coefficient correlation',
                         '8-1 Angular second moment', '8-2 Contrast', '8-3 Correlation', '8-4 Sum of squares', '8-5 Inverse difference moment', '8-6 Sum average', '8-7 Sum variance', '8-8 Sum entropy', '8-9 Entropy', '8-10 Difference variance', '8-11 Diference entropy', '8-12 Information correlation','8-13 Coefficient correlation', 
                         '9-1 Angular second moment', '9-2 Contrast', '9-3 Correlation', '9-4 Sum of squares', '9-5 Inverse difference moment', '9-6 Sum average', '9-7 Sum variance', '9-8 Sum entropy', '9-9 Entropy', '9-10 Difference variance', '9-11 Diference entropy', '9-12 Information correlation','9-13 Coefficient correlation',
                         '10-1 Angular second moment', '10-2 Contrast', '10-3 Correlation', '10-4 Sum of squares', '10-5 Inverse difference moment', '10-6 Sum average', '10-7 Sum variance', '10-8 Sum entropy', '10-9 Entropy', '10-10 Difference variance', '10-11 Diference entropy', '10-12 Information correlation','10-13 Coefficient correlation',
                         '11-1 Angular second moment', '11-2 Contrast', '11-3 Correlation', '11-4 Sum of squares', '11-5 Inverse difference moment', '11-6 Sum average', '11-7 Sum variance', '11-8 Sum entropy', '11-9 Entropy', '11-10 Difference variance', '11-11 Diference entropy', '11-12 Information correlation','11-13 Coefficient correlation', 
                         '12-1 Angular second moment', '12-2 Contrast', '12-3 Correlation', '12-4 Sum of squares', '12-5 Inverse difference moment', '12-6 Sum average', '12-7 Sum variance', '12-8 Sum entropy', '12-9 Entropy', '12-10 Difference variance', '12-11 Diference entropy', '12-12 Information correlation','12-13 Coefficient correlation',
                         '13-1 Angular second moment', '13-2 Contrast', '13-3 Correlation', '13-4 Sum of squares', '13-5 Inverse difference moment', '13-6 Sum average', '13-7 Sum variance', '13-8 Sum entropy', '13-9 Entropy', '13-10 Difference variance', '13-11 Diference entropy', '13-12 Information correlation','13-13 Coefficient correlation']

  
  if(axis0 > 1 and axis1 > 1 and axis2 > 1):
    les40 = pd.DataFrame()  
    les41 = pd.Series()
    
    for v in range(0, 13):     
      for w in range(0, 13):
        les41 = np.concatenate((les41, har_box[v][w]), axis=None)
        les41=pd.Series(les41)
           
      haralick3d_dict2 =  {f'h_{haralick_lesion_3d[i]}': val for i, val in enumerate(har_box[v])}    
      haralick_lesion_3d_series = pd.Series({**haralick3d_dict2}, name=f'Lesion_{roi_i}_{v}')
      lesion_3d_features = lesion_3d_features.append(haralick_lesion_3d_series)  
      subject_features_haralick_d1 =  subject_features_haralick_d1.append(haralick_lesion_3d_series, ignore_index=False)

    les40 = les.append(les41, ignore_index=True)
    les42 = les42.append(les40)
    haralick3d_dict = {f'h_{haralick_lesion_3d[i]}': val for i, val in enumerate(h_feature.mean(axis=0))}
  
  else:
    h_feature = {}
    haralick3d_dict = {f'h_{haralick_lesion_3d[i]}': 0 for i, val in enumerate(haralick_lesion_3d)}

  print('2. Haralicks: BOX')
  print(display(HTML(les40.to_html())))
  print(display(HTML(lesion_3d_features.to_html())))
  


### HARALICK ROI

  lesion_3d_features = pd.DataFrame()
  haralick_lesion_3d = ['1 Angular second moment', '2 Contrast', '3 Correlation', '4 Sum of squares', '5 Inverse difference moment', '6 Sum average', '7 Sum variance', '8 Sum entropy', '9 Entropy', '10 Difference variance', '11 Diference entropy', '12 Information correlation','13 Coefficient correlation']
  haralick_lesion_3d0= ['1-1 Angular second moment', '1-2 Contrast', '1-3 Correlation', '1-4 Sum of squares', '1-5 Inverse difference moment', '1-6 Sum average', '1-7 Sum variance', '1-8 Sum entropy', '1-9 Entropy', '1-10 Difference variance', '1-11 Diference entropy', '1-12 Information correlation','1-13 Coefficient correlation',
                         '2-1 Angular second moment', '2-2 Contrast', '2-3 Correlation', '2-4 Sum of squares', '2-5 Inverse difference moment', '2-6 Sum average', '2-7 Sum variance', '2-8 Sum entropy', '2-9 Entropy', '2-10 Difference variance', '2-11 Diference entropy', '2-12 Information correlation','2-13 Coefficient correlation', 
                         '3-1 Angular second moment', '3-2 Contrast', '3-3 Correlation', '3-4 Sum of squares', '3-5 Inverse difference moment', '3-6 Sum average', '3-7 Sum variance', '3-8 Sum entropy', '3-9 Entropy', '3-10 Difference variance', '3-11 Diference entropy', '3-12 Information correlation','3-13 Coefficient correlation',
                         '4-1 Angular second moment', '4-2 Contrast', '4-3 Correlation', '4-4 Sum of squares', '4-5 Inverse difference moment', '4-6 Sum average', '4-7 Sum variance', '4-8 Sum entropy', '4-9 Entropy', '4-10 Difference variance', '4-11 Diference entropy', '4-12 Information correlation','4-13 Coefficient correlation',
                         '5-1 Angular second moment', '5-2 Contrast', '5-3 Correlation', '5-4 Sum of squares', '5-5 Inverse difference moment', '5-6 Sum average', '5-7 Sum variance', '5-8 Sum entropy', '5-9 Entropy', '5-10 Difference variance', '5-11 Diference entropy', '5-12 Information correlation','5-13 Coefficient correlation', 
                         '6-1 Angular second moment', '6-2 Contrast', '6-3 Correlation', '6-4 Sum of squares', '6-5 Inverse difference moment', '6-6 Sum average', '6-7 Sum variance', '6-8 Sum entropy', '6-9 Entropy', '6-10 Difference variance', '6-11 Diference entropy', '6-12 Information correlation','6-13 Coefficient correlation',
                         '7-1 Angular second moment', '7-2 Contrast', '7-3 Correlation', '7-4 Sum of squares', '7-5 Inverse difference moment', '7-6 Sum average', '7-7 Sum variance', '7-8 Sum entropy', '7-9 Entropy', '7-10 Difference variance', '7-11 Diference entropy', '7-12 Information correlation','7-13 Coefficient correlation',
                         '8-1 Angular second moment', '8-2 Contrast', '8-3 Correlation', '8-4 Sum of squares', '8-5 Inverse difference moment', '8-6 Sum average', '8-7 Sum variance', '8-8 Sum entropy', '8-9 Entropy', '8-10 Difference variance', '8-11 Diference entropy', '8-12 Information correlation','8-13 Coefficient correlation', 
                         '9-1 Angular second moment', '9-2 Contrast', '9-3 Correlation', '9-4 Sum of squares', '9-5 Inverse difference moment', '9-6 Sum average', '9-7 Sum variance', '9-8 Sum entropy', '9-9 Entropy', '9-10 Difference variance', '9-11 Diference entropy', '9-12 Information correlation','9-13 Coefficient correlation',
                         '10-1 Angular second moment', '10-2 Contrast', '10-3 Correlation', '10-4 Sum of squares', '10-5 Inverse difference moment', '10-6 Sum average', '10-7 Sum variance', '10-8 Sum entropy', '10-9 Entropy', '10-10 Difference variance', '10-11 Diference entropy', '10-12 Information correlation','10-13 Coefficient correlation',
                         '11-1 Angular second moment', '11-2 Contrast', '11-3 Correlation', '11-4 Sum of squares', '11-5 Inverse difference moment', '11-6 Sum average', '11-7 Sum variance', '11-8 Sum entropy', '11-9 Entropy', '11-10 Difference variance', '11-11 Diference entropy', '11-12 Information correlation','11-13 Coefficient correlation', 
                         '12-1 Angular second moment', '12-2 Contrast', '12-3 Correlation', '12-4 Sum of squares', '12-5 Inverse difference moment', '12-6 Sum average', '12-7 Sum variance', '12-8 Sum entropy', '12-9 Entropy', '12-10 Difference variance', '12-11 Diference entropy', '12-12 Information correlation','12-13 Coefficient correlation',
                         '13-1 Angular second moment', '13-2 Contrast', '13-3 Correlation', '13-4 Sum of squares', '13-5 Inverse difference moment', '13-6 Sum average', '13-7 Sum variance', '13-8 Sum entropy', '13-9 Entropy', '13-10 Difference variance', '13-11 Diference entropy', '13-12 Information correlation','13-13 Coefficient correlation']

  
  if(axis0 > 1 and axis1 > 1 and axis2 > 1):
    les60 = pd.DataFrame()  
    les61 = pd.Series()
    
    for v in range(0, 13):     
      for w in range(0, 13):
        les61 = np.concatenate((les61, har_roi[v][w]), axis=None)
        les61 = pd.Series(les61)
           
      haralick3d_dict2 =  {f'h_{haralick_lesion_3d[i]}': val for i, val in enumerate(har_roi[v])}    
      haralick_lesion_3d_series = pd.Series({**haralick3d_dict2}, name=f'Lesion_{roi_i}_{v}')
      lesion_3d_features = lesion_3d_features.append(haralick_lesion_3d_series)  
      subject_features_haralick_d1 =  subject_features_haralick_d1.append(haralick_lesion_3d_series, ignore_index=False)

    les60 = les60.append(les61, ignore_index=True)
    les62 = les62.append(les60)
    haralick3d_dict = {f'h_{haralick_lesion_3d[i]}': val for i, val in enumerate(h_feature.mean(axis=0))}
  
  else:
    h_feature = {}
    haralick3d_dict = {f'h_{haralick_lesion_3d[i]}': 0 for i, val in enumerate(haralick_lesion_3d)}

  print('2. Haralicks: ROI')
  print(display(HTML(les60.to_html())))
  print(display(HTML(lesion_3d_features.to_html())))
  lesion_3d_features30 = lesion_3d_features30.append(les60)




  Lesion10= Lesion1[0]
  mean= np.mean(Lesion10)


  plt.figure()
  plt.imshow(Lesion10, cmap='gray')
  plt.xticks(np.arange(-.5, axis2, 1))
  plt.yticks(np.arange(-.5, axis1, 1))
  plt.grid(True)
  for i in range(axis1):
    for j in range(axis2):
      if Lesion10[i,j] > mean:
        plt.text(j, i, Lesion10[i,j], ha="center", va="center", color="black")
      else:
        plt.text(j, i, Lesion10[i,j], ha="center", va="center", color="w")     
  #plt.colorbar()
  plt.title('Lesion slide 2D')
  plt.show()

  Lesion100= Lesion202
  mean1 = np.max(Lesion100)/2

  plt.figure()
  plt.imshow(Lesion100, cmap='viridis')
  plt.xticks(np.arange(-.5, 7.5, 1))
  plt.yticks(np.arange(-.5, 7.5, 1))
  plt.grid(True)
  for i1 in range(max):
    for j1 in range(max):
      if Lesion100[i1,j1] > mean1 :
        plt.text(j1, i1, Lesion100[i1,j1], ha="center", va="center", color="black")
      else:
        plt.text(j1, i1, Lesion100[i1,j1], ha="center", va="center", color="w")
  plt.title('GLCM 1 Horizontal')
  plt.xlabel('i')
  plt.ylabel('j')
  plt.show()

  print(np.sum(Lesion100))




  Lesion1110= Lesion202 / Lesion202.sum()


  plt.figure()
  plt.imshow(Lesion1110, cmap='viridis')
  plt.xticks(np.arange(-.5, 7.5, 1))
  plt.yticks(np.arange(-.5, 7.5, 1))
  plt.grid(True)
  for i1 in range(max):
        for j1 in range(max):
          if Lesion1110[i1,j1] > np.mean(Lesion1110):
            plt.text(j1, i1, round(Lesion1110[i1,j1],3), ha="center", va="center", color="black")
          else:
            plt.text(j1, i1, round(Lesion1110[i1,j1],3), ha="center", va="center", color="w")
  plt.title('GLCM 1 Horizontal (Norm)')
  plt.xlabel('i')
  plt.ylabel('j')
  plt.show()

  
print(display(HTML(lesion_3d_features30.to_html())))
  
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
