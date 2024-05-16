### Setup ###
import glob

import numpy as np
from matplotlib import pyplot as plt
from skimage import io, img_as_ubyte, morphology
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters.edges import prewitt_h
import SimpleITK as sitk
from skimage.measure import label, regionprops
import pydicom as dicom
from skimage.morphology import disk, closing, opening, binary_closing, binary_opening
from skimage.transform import SimilarityTransform
from sklearn import decomposition
from scipy.spatial import distance


### Question 1 ###
def question1():
    # Size of one frame
    image_size = 2400 * 1200 * 3

    # 35 megabytes in bytes
    max_size = 35 * 1000 * 1000

    # How many frames per second can be transfered?
    fps_transfer = max_size / image_size

    # How many frames can be analyzed in one second?
    fps_analyze = 1 / 0.130

    print(fps_transfer)
    print(fps_analyze)


### Question 2-4 ###
def question2_4():
    in_dir = "data/Pixelwise/"

    # Load the image ardeche_river.jpg
    ardeche_river = io.imread(in_dir + "ardeche_river.jpg")

    # Convert to grayscale
    ardeche_river_gray = rgb2gray(ardeche_river)

    # Perform a linear grayscale histogram stretch where the minimum value is 0.2 and the maximum is 0.8
    d_min = 0.2
    d_max = 0.8
    ardeche_river_gray_stretch = ((d_max - d_min) / (np.max(ardeche_river_gray) - np.min(ardeche_river_gray))
                                  * (ardeche_river_gray - np.min(ardeche_river_gray)) + d_min)

    # Print the minimum and maximum values of the stretched image to check if the stretch was successful
    # print(ardeche_river_gray_stretch.min())
    # print(ardeche_river_gray_stretch.max())

    # Compute the average value of the stretched image
    avg_value = np.mean(ardeche_river_gray_stretch)

    # Use prewitt_h to extract edges
    prewitt_h_ardeche = prewitt_h(ardeche_river_gray_stretch)

    # Compute the maximum absolute value of the edges
    max_edge = np.max(np.abs(prewitt_h_ardeche))

    # Create a binary mask by using the average value as a threshold
    mask = ardeche_river_gray_stretch > avg_value

    # Count the number of pixels in the mask
    num_pixels = np.sum(mask)

    # Print the results
    print(f"The average pixel value is: {avg_value}")
    print(f"The maximum edge value is: {max_edge}")
    print(f"The number of pixels in the mask is: {num_pixels}")


### Question 5 ###
def rotation_matrix(pitch, roll, yaw, deg=False):
    """
    Return the rotation matrix associated with the Euler angles roll, pitch, yaw.

    Parameters
    ----------
    pitch : float
        The rotation angle around the x-axis.
    roll : float
        The rotation angle around the y-axis.
    yaw : float
        The rotation angle around the z-axis.
    deg : bool, optional
        If True, the angles are given in degrees. If False, the angles are given
        in radians. Default: False.
    """
    if deg:
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)

    R_x = np.array([[1, 0, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch), 0],
                    [0, np.sin(pitch), np.cos(pitch), 0],
                    [0, 0, 0, 1]])

    R_y = np.array([[np.cos(roll), 0, np.sin(roll), 0],
                    [0, 1, 0, 0],
                    [-np.sin(roll), 0, np.cos(roll), 0],
                    [0, 0, 0, 1]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],
                    [np.sin(yaw), np.cos(yaw), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    R = np.dot(np.dot(R_x, R_y), R_z)

    return R


def imshow_orthogonal_view(sitkImage, origin=None, title=None):
    """
    Display the orthogonal views of a 3D volume from the middle of the volume.

    Parameters
    ----------
    sitkImage : SimpleITK image
        Image to display.
    origin : array_like, optional
        Origin of the orthogonal views, represented by a point [x,y,z].
        If None, the middle of the volume is used.
    title : str, optional
        Super title of the figure.

    Note:
    On the axial and coronal views, patient's left is on the right
    On the sagittal view, patient's anterior is on the left
    """
    data = sitk.GetArrayFromImage(sitkImage)

    if origin is None:
        origin = np.array(data.shape) // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    data = img_as_ubyte(data / np.max(data))
    axes[0].imshow(data[origin[0], ::-1, ::-1], cmap='gray')
    axes[0].set_title('Axial')

    axes[1].imshow(data[::-1, origin[1], ::-1], cmap='gray')
    axes[1].set_title('Coronal')

    axes[2].imshow(data[::-1, ::-1, origin[2]], cmap='gray')
    axes[2].set_title('Sagittal')

    [ax.set_axis_off() for ax in axes]

    if title is not None:
        fig.suptitle(title, fontsize=16)

    plt.show()


def question5_7():
    in_dir = "data/ImageRegistration/"

    t1_v1 = sitk.ReadImage(in_dir + "ImgT1_v1.nii.gz")
    t1_v2 = sitk.ReadImage(in_dir + "ImgT1_v2.nii.gz")

    # Affine matrix A
    A = np.array([[0.98, -0.16, 0.17, 0],
                  [0.26, 0.97, 0, -15],
                  [-0.17, 0.04, 0.98, 0],
                  [0, 0, 0, 1]])

    # Extract the rotation/scaling matrix and the translation array
    rotation_scaling_matrix = A[:3, :3]
    translation = A[:3, 3]
    print(translation)
    translation = [0, -15, 0]

    centre_image = np.array(t1_v1.GetSize()) / 2 - 0.5  # Image Coordinate System
    centre_world = t1_v1.TransformContinuousIndexToPhysicalPoint(centre_image)  # World Coordinate System

    # Create the affine transformation
    transform = sitk.AffineTransform(3)
    transform.SetCenter(centre_world)
    transform.SetMatrix(rotation_scaling_matrix.flatten())
    transform.SetTranslation(translation)

    # Apply the transformation to the image t1_v1
    ImgT1_v2 = sitk.Resample(t1_v2, transform)

    # Convert the transformed SimpleITK image to a numpy array
    ImgT1_v2_array = sitk.GetArrayFromImage(ImgT1_v2)

    # Display the orthogonal view
    imshow_orthogonal_view(ImgT1_v2, title='ImgT1_v1')

    fixed_image = t1_v1
    moving_image = t1_v2

    # Set the registration - Fig. 1 from the Theory Note
    R = sitk.ImageRegistrationMethod()

    # Set a one-level the pyramid scheule. [Pyramid step]
    R.SetShrinkFactorsPerLevel(shrinkFactors=[2])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Set the interpolator [Interpolation step]
    R.SetInterpolator(sitk.sitkLinear)

    # Set the similarity metric [Metric step]
    R.SetMetricAsMeanSquares()

    # Set the sampling strategy [Sampling step]
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.10)

    # Set the optimizer [Optimization step]
    R.SetOptimizerAsPowell(stepLength=0.1, numberOfIterations=25)

    # Initialize the transformation type to rigid
    initTransform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler3DTransform(),
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R.SetInitialTransform(initTransform, inPlace=False)

    # Some extra functions to keep track to the optimization process
    # R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R)) # Print the iteration number and metric value

    # Estimate the registration transformation [metric, optimizer, transform]
    tform_reg = R.Execute(fixed_image, moving_image)

    # Apply the estimated transformation to the moving image
    ImgT1_B = sitk.Resample(moving_image, tform_reg)

    params = tform_reg.GetParameters()
    angles = params[:3]
    trans = params[3:6]
    print('Estimated translation: ')
    print(np.round(trans, 3))
    print('Estimated rotation (deg): ')
    print(np.round(np.rad2deg(angles), 3))

    angle = -20  # degrees
    roll_radians = np.deg2rad(angle)
    rot_matrix = rotation_matrix(0, roll_radians, 0)[:3, :3]  # SimpleITK inputs the rotation and the translation separately

    # Create the Affine transform and set the rotation
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(rot_matrix.T.flatten())

    # centre_image = np.array(movingImage.GetSize()) / 2 - 0.5  # Image Coordinate System
    # centre_world = movingImage.TransformContinuousIndexToPhysicalPoint(centre_image) # World Coordinate System

    # Apply the transformation to the image
    movingImage_reg = sitk.Resample(moving_image, transform)
    imshow_orthogonal_view(movingImage_reg, title='Moving image')

    mask = sitk.GetArrayFromImage(fixed_image) > 50
    fixedImageNumpy = sitk.GetArrayFromImage(fixed_image)
    movingImageNumpy = sitk.GetArrayFromImage(movingImage_reg)

    fixedImageVoxels = fixedImageNumpy[mask]
    movingImageVoxels = movingImageNumpy[mask]
    mse = np.mean((fixedImageVoxels - movingImageVoxels) ** 2)
    print('Anwer: MSE = {:.2f}'.format(mse))


### Question 8-10 ###
def question8_10():
    in_dir = "data/ChangeDetection/"

    # Load the images
    frame_1 = io.imread(in_dir + "frame_1.jpg")
    frame_2 = io.imread(in_dir + "frame_2.jpg")

    # Convert to HSV
    frame_1_hsv = rgb2hsv(frame_1)
    frame_2_hsv = rgb2hsv(frame_2)

    # Extract the saturation channel and scale them with 255
    frame_1_saturation = frame_1_hsv[:, :, 1] * 255
    frame_2_saturation = frame_2_hsv[:, :, 1] * 255

    # Compute the absolute difference between the saturation channels
    diff_saturation = np.abs(frame_1_saturation - frame_2_saturation)

    # Compute the mean and standard deviation of the difference
    mean_diff = np.mean(diff_saturation)
    std_diff = np.std(diff_saturation)

    # Compute the threshold as the mean plus two times the standard deviation
    threshold = mean_diff + (2 * std_diff)

    # Create a binary mask by thresholding the difference image with the threshold,
    # setting the mask to 1 where the difference is larger than the threshold
    mask = diff_saturation
    mask[mask > threshold] = 1

    # Count the number of pixels in the mask that are equal to 1
    num_pixels = np.sum(mask == 1)

    # Perform BLOB analysis on the binary mask
    label_image = label(diff_saturation, connectivity=2)

    # Find the BLOB with the largest area
    regions = regionprops(label_image)
    max_area = 0
    for region in regions:
        if region.area > max_area:
            max_area = region.area

    # Print the results
    print(f"The largest area is: {max_area}")
    print(f"The threshold is: {threshold}")
    print(f"The number of pixels in the mask is: {num_pixels}")


### Question 11 ###
def hough(x, y, theta):
    return x * np.cos(np.deg2rad(theta)) + y * np.sin(np.deg2rad(theta))


def question11():
    print(f"Left arm: {225-180}, {hough(3, 5, (225-180)):.1f}")
    print(f"Right arm: {135-180}, {hough(3, 5, 135-180):.1f}")
    print(f"Head: 0, {hough(3, 6, 0):.1f}")
    print(f"Left leg: {225-180}, {hough(3, 2, (225-180)):.1f}")
    print(f"Right leg: {135-180}, {hough(3, 2, (135-180)):.1f}")

    print(f"\n\n\n")

    print(f"Left arm: 45, {hough(2, 4, 45):.2f}")
    print(f"Right arm: {315 - 360}, {hough(4, 4, 315 - 360):.2f}")
    print(f"Head: 0, {hough(3, 6, 0):.2f}")
    print(f"Left leg: 45, {hough(1, 0, 45):.2f}")
    print(f"Right leg: {315 - 360}, {hough(5, 0, 315 - 360):.2f}")


### Question 12-15 ###
def question12_15():
    in_dir = "data/HeartCT/"

    # Load the 1-001.dcm
    dcm = dicom.dcmread(in_dir + "1-001.dcm")

    # Load the BloodROI.png and MyocardiumROI.png
    blood_roi = io.imread(in_dir + "BloodROI.png")
    myocardium_roi = io.imread(in_dir + "MyocardiumROI.png")

    # Load the ground truth, BllodGT.png
    blood_gt = io.imread(in_dir + "BloodGT.png")

    # Extract the values from the roi images
    blood_values = dcm.pixel_array[blood_roi == 1]
    myocardium_values = dcm.pixel_array[myocardium_roi == 1]

    # Compute the mean and standard deviation of the blood values
    blood_mean = np.mean(blood_values)
    blood_std = np.std(blood_values)

    # Create a binary mask by thresholding the image
    T1 = blood_mean - 3 * blood_std
    T2 = blood_mean + 3 * blood_std
    segmented = (dcm.pixel_array > T1) & (dcm.pixel_array < T2)

    # Perform a morphological closing operation on the binary mask with a disk of radius 3
    segmented = binary_closing(segmented, disk(3))

    # Perform a morphological opening operation on the binary mask with a disk of radius 5
    segmented = binary_opening(segmented, disk(5))

    # Perform a connected component analysis on the binary mask
    label_image = label(segmented, connectivity=2)

    # Compute the area of the BLOB's
    regions = regionprops(label_image)

    # Only keep BLOB's with an area larger than 2000 pixels and smaller than 5000 pixels
    filtered_regions = [region for region in regions if 2000 < region.area < 5000]

    # Filter the BLOB's on the label_image
    for region in regions:
        if region not in filtered_regions:
            label_image[label_image == region.label] = 0

    gt_bin = blood_gt > 0
    i_blood = label_image > 0
    dice = 2 * np.sum(segmented * blood_gt) / (np.sum(segmented) + np.sum(blood_gt))
    dice_score = 1 - distance.dice(i_blood.ravel(), gt_bin.ravel())

    # Print the class range, the number of BLOB's, and the Dice similarity coefficient
    print(f"The class range is: {blood_mean - 3 * blood_std:.0f} to {blood_mean + 3 * blood_std:.0f}")
    print(f"The number of BLOB's is: {len(regions)}")
    print(f"The Dice similarity coefficient is: {dice}")
    print(f"The real Dice similarity coefficient is: {dice_score:.2f}")

    # Compute the minimum distance classifier
    myocardium_mean = np.mean(myocardium_values)

    # Compute the threshold
    threshold = (blood_mean + myocardium_mean) / 2

    # Print the threshold
    print(f"The threshold is: {threshold:.0f}")


### Question 16-19 ###
def question16_19():
    in_dir = "data/pistachio/"

    # Load the txt pistachio_data.txt
    pistachio_data = np.loadtxt(in_dir + "pistachio_data.txt", comments="%")

    # Extract the features and labels
    x = pistachio_data[0:200, 0:12]

    # Subtract mean and divide by standard deviation
    mn = np.mean(x, axis=0)
    data = x - mn
    data = data / data.std(axis=0)

    # PCA
    pca = decomposition.PCA()
    pca.fit(data)
    values_pca = pca.explained_variance_
    exp_var = pca.explained_variance_ratio_
    vectors_pca = pca.components_

    data_transform = pca.transform(data)

    # Compute the sum of squared projected values for the first nut
    sum_squared = np.sum(data_transform[0] ** 2)

    # Print the sum of squared projected values
    print(f"The sum of squared projected values is: {sum_squared:.1f}")

    # Print the feature with the smallest standard deviation
    temp_data = x - mn
    std = temp_data.std(axis=0)
    min_std_idx = np.argmin(std)
    print(f"The feature with the smallest standard deviation is: {min_std_idx}")
    # The fourth feature is Eccentricity

    # How many components are needed to explain at least 97% of the total variation
    sum_var = 0
    for i, var in enumerate(exp_var):
        sum_var += var
        if sum_var >= 0.97:
            print(f"The number of components needed to explain at least 97% of the total variation is: {i + 1}")
            break

    # Compute the covariance matrix
    C_X_np = np.cov(data, rowvar=False)

    # Print the maximum absolute value in the covariance matrix
    print(f"The maximum absolute value in the covariance matrix is: {np.max(np.abs(C_X_np)):.2f}")


### Question 20-22 ###
def compute_alignment_error(source, destination):
    distances = np.sum((source - destination) ** 2, axis=1)
    error = np.sum(distances)
    return error


def question20_22():
    src = np.array([[3, 1], [3.5, 3], [4.5, 6], [5.5, 5], [7, 1]])
    dst = np.array([[1, 0], [2, 4], [3, 6], [4, 4], [5, 0]])

    # Calculate the alignment error F
    F = compute_alignment_error(src, dst)

    mean_src = np.mean(src, axis=0)
    mean_dst = np.mean(dst, axis=0)

    tform = SimilarityTransform()
    tform.estimate(src, dst)

    print(f"The optimal translation is: {mean_dst - mean_src}")
    print(f"The alignment error F is: {F:.2f}")
    print(f"The optimal rotation is: {np.rad2deg(np.abs(tform.rotation)):.0f}")


### Question 23-25 ###
def create_u_byte_image_from_vector(im_vec, height, width, channels):
    min_val = im_vec.min()
    max_val = im_vec.max()

    # Transform to [0, 1]
    im_vec = np.subtract(im_vec, min_val)
    im_vec = np.divide(im_vec, max_val - min_val)
    im_vec = im_vec.reshape(height, width, channels)
    im_out = img_as_ubyte(im_vec)
    return im_out


def question23_25():
    in_dir = "data/Fish/"

    all_images = glob.glob(in_dir + "*.jpg")
    n_samples = len(all_images)

    im_org = io.imread(all_images[0])
    im_shape = im_org.shape
    height = im_shape[0]
    width = im_shape[1]
    channels = im_shape[2]
    n_features = height * width * channels

    data_matrix = np.zeros((n_samples, n_features))

    # Load all images and store them in the data_matrix
    idx = 0
    for image_file in all_images:
        img = io.imread(image_file)
        flat_img = img.flatten()
        data_matrix[idx, :] = flat_img
        idx += 1

    # Compute the mean of each row
    average_fish = np.mean(data_matrix, axis=0)

    # Display the average fish
    # mean_fish_img = create_u_byte_image_from_vector(average_fish, height, width, channels)
    # io.imshow(mean_fish_img)
    # io.show()

    # PCA
    pca = decomposition.PCA(n_components=6)
    pca.fit(data_matrix)

    # Print the total variance explained by the first two principal components
    print(f"Total variance explained by the first two principal components: "
          f"{np.sum(pca.explained_variance_ratio_[:2]) * 100:.0f}%")

    # Load the image neon.jpg and the image guppy.jpg
    neon = io.imread(in_dir + "neon.jpg")
    guppy = io.imread(in_dir + "guppy.jpg")

    # Compute the pixelwise sum of squared differences between the two images
    ssd = np.sum((neon - guppy) ** 2)

    # Print the sum of squared differences
    print(f"The sum of squared differences between the two images is: {ssd:.0f}")

    transformed_data = pca.transform(data_matrix)

    # Find the fish furthest away from the neon fish in PCA space
    neon_transformed = pca.transform(neon.flatten().reshape(1, -1))
    distances = np.sum((transformed_data - neon_transformed) ** 2, axis=1)
    max_idx = np.argmax(distances)

    # Print the index of the fish furthest away from the neon fish
    print(f"The fish furthest away from the neon fish is: {max_idx} : {all_images[max_idx]}")


### Main ###

if __name__ == '__main__':
    # question1()
    # question2_4()
    question5_7()
    # question8_10()
    # question11()
    # question12_15()
    # question16_19()
    # question20_22()
    # question23_25()
