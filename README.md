# TipQUANT

Automated tool for retrieving data, such as tip location and membrane intensity distribution, from pollen growth videos.
![tube_raw](data/assets/tube_raw.png)
![tube_raw](data/assets/tube_processed.png)

This project is open source and developed in python.

## Installation

### Linux, MacOS

To install TipQUANT on your machine you will need python 3.6+.

- Getting the code from the repository

```sh
cd $CODE_DIRECTORY
git clone <todo insert github url>
```

- Setting up the python environment (use of virtualenv recommended)

```sh
python3 -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
(venv) cd $CODE_DIRECTORY
(venv) pip install -r requirements.txt
```

- You will also need to have ffmpeg installed

```sh
brew install ffmpeg  # mac
sudo apt install ffmpeg  # ubuntu
```

- Run the application

```sh
(venv) python run_app.py
```

### Specific instructions for Mac M1

On the latest Mac (as of October 2021), some packages are not compatible yet with the new CPU from Apple. You might need to install an x86_64 compiled version of Python in order to install all the packages. To do so, you can check out this [tutorial](https://julianlegouic.github.io/setup-MacM1-x86_64/) (only in French) or directly the source of the tutorial [here](https://stackoverflow.com/questions/68659865/cannot-pip-install-mediapipe-on-macos-m1) (until step 7 included).

### Windows

The easiest way to install this application is to use Anaconda (<https://docs.anaconda.com/anaconda/install/windows/>).

- Getting the code from the repository (with github it is easier to use git bash command line tool)

```sh
cd $CODE_DIRECTORY
git clone <insert github url>
```

- Setting up the python environement

```sh
cd $CODE_DIRECTORY
conda env create --file environment.yml
conda activate tip_quant
```

- You will also need to have ffmpeg installed (builds available at <https://github.com/BtbN/FFmpeg-Builds/releases>).
- Remember to add the bin folder to your windows `PATH` (see section 3 <http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/>)
- Run the application

```sh
(tip_quant) python run_app.py
```

## How to use it ?

- The side bar on the left is dedicated to user input parameters
![sidebar](data/assets/sidebar%20-%20copie.png)
- You can load your video on the main panel
![upload_video](data/assets/upload_video.png)
- Once you have uploaded your video and picked your parameters, you can click on the "Run" button, at the end of the side bar. Data will be saved locally in the path specified in the "Output directory" field (`data/output` by default).
![run_button](data/assets/run_button.png)

### Command-line

We provide a command-line tool for convenience. To use it, activate your virtual environment (with the required packages installed) and run the command-line script:

```sh
source /path/to/new/virtual/environment/bin/activate
(venv) python run_cli.py -c /path/to/config.json
```

An example config file is provided under `src/config.json`.

The config file defines the same parameters as the GUI does. Be careful to setting the `OUTPUT_DIR` and `VIDEO_PATH` parameters properly.

## Model assumptions

Our model is based on two key assumptions:

- There is one only pollen tube in the video
- The pollen tube cannot grow towards multiple direction at a given time

## How does it work ?

As a reminder, our work aims to automatically find the tip, defined by the point on the contour with maximal growth on the pollen tube, so that we can derive useful measurements from it.

### Code organization

From the code perspective, we have one data holder class: `PollenTube` in `src/core/model.py`, that fully describes all the characteristics (such as contour, normals, tip, ...) of a pollen tube.

A tube is instantiated "empty" (every attribute set to `None`) and is filled by the different modules in `src/core/contour.py` and `src/core/tip.py`. The modules are classes that serve only one purpose (for example, detecting the contour or fitting splines). Their attributes are the parameters of the model.

There are also modules in `src/core/region.py`, that define the measurement areas in the file. And a module for measurements, located in `src/core/measure.py`, which provides methods to compute the output of this application.

#### Notes on object representation

- A `contour` is represented by a `numpy.array` of size `n_contour_points * 2`. One key thing is that, points are sorted by adjacency in a counter-clockwise manner.
- Normals and tangents are also `numpy.array` of size `n_contour_points * 2`. Objects tied to the contour should have the same length as the contour. It is also the case for `displacements` and `curvs` (curvatures), though they are scalars so their dimension is `n_contour_points * 1`.
- Every variable mentionning `indices` is referring to the indices wrt a contour. For instance, to get the tip location from `tip_index`, we use `contour[tip_index]`.

### Main pipeline

1. We start with a list of `frames` and `config` parameters (user defined). The `config` allows us to instantiate every module.

2. For each frame:
    1. We detect the principal contour after preprocessing the frames, using the `ContourDetection` module.
    2. We fit splines to the contour using the `ContourParameterization` module; splines are useful to reduce noise in the computations of normals and tangents vectors (we could otherwise use finite differences).
    3. We compute tangents by derivating the fitted splines. Normals by rotating tangents and normalizing the result. And curvatures by taking the norm of the tangents derivatives. This is done by the `ContourCharacterization` module.
    4. We save the `contour` detected at this step to help the next contour detection: as the intensity inside the pollen tube varies, the detected contour at time `t+1` is sometimes smaller than the contour at time `t`. This is just an artifact from contour detection, so we ensure that the `contour` never decreases by computing the union of the detected contours from one step to another.
    5. We save the result of the iteration in the `PollenTube` model.

    At this point, we have, for each frame, a good contour representation as well as some of its properties like normals. We can then locate the tip by iterating over the contours and looking one step ahead. (This allow us to compute growth and the so-called tip.)

3. For each contour:
    1. We compute its region of interest (or of growth) by substracting the mask of the future contour at time `t + step` with the current mask at time `t`. This is handled by the `ContourROI` module.
    2. For each point in the region of interest, we compute its displacement, defined by the length of the intersection between its normal vector and the future contour at time `t+step` (`ContourDisplacement` module). ![displacement](data/assets/displacement.png)
    3. We compute the main growth direction as the sum of the normal vectors from the roi points weighted by their displacementn (still `ContourDisplacement` module).
    4. Finally, we locate the tip by sliding a window over the region of interest points with the `TipDetection` module. For each point in the window, we compute the dot product of its normal, weighted by the exponential 2 of its displacement, and the main growth direction, and then compute the average of all the dot products over the window. The window with the maximum value is selected and its center is defined as the tip location.
    5. We save the result of the iteration in the `PollenTube` model.

4. We run the clustering algorithm DBSCAN to detect wrongly placed tips. Wrong tips are adjusted from the valid ones. This is done using the `TipCorrection` class. The need for correction happens in some cases where there are a lot of noise around the pollen tube or if the pollen tube contour is not good enough.

    We now possess for every frames (except the last `step` ones), their contours, properties and tips. We can know perform measures.

5. For each tube:
    1. We compute the membrane area by extending an arc of user-defined length and width around the tip with the `Membrane` module.
    2. We compute the region of measurement inside the cytoplasm (non-overlapping with the membrane) with the `Region <A, B, C or D>` module.
    3. We use the `Measure` module to compute useful figures from the masks above. A list of the measurements is described below. ![areas](data/assets/areas.png)

6. We save the results and return them to the user.

## Parameters description

Here we will describe every parameters of the model, though some are more sensitive than others (see next section).

### Video parameters (mandatory)

- **pixel size**: the size of one pixel in micrometers
- **timestep**: the time delta between two frames in seconds

### Contour detection

- **kernel size**: the size of the kernel used for preprocessing operations (gaussian blur, dilation and erosion) of the original frames
- **sigma**: the standard deviation used in the gaussian blur
- **gamma**: the gamma coefficient used in the gamma correction filter in preprocessing

### Splines

- **number of knots**: number of contour points that will be considered as knot points. A float
value between 0 and 1 corresponds to a percentage of the number of points in the contour; or an
integer that will be the number of knots
- **degree** : splines degree

### Tip detection

- **window size**: size of the window used to average contour points characteristics to find the tip. Larger values will cancel noise but will not detect well very localized growth.
- **step**: number of frames used to look in the future, this is used to determine growth direction. Fast changing videos require a lesser step.

### Tip correction

Below parameters express the expected number of tips around a given tip. As the pollen tube grows, the tip is supposed to move relatively smoothly from one frame to another. Thus, if we look at the full picture (all the detected tips on the 2D plane), we expect a "chain" of tips. For one particular tip, the previous ones and next ones should be close, depending on the speed of growth.

- **epsilon**: distance (um) to look from one tip for its neighbours
- **samples**: number of tips expected in the epsilon distance

### Membrane

- **length**: length of the membrane in micrometers
- **thickness**: thickness of the membrane in pixel units

### Cytoplasm region

- **type**: shape of the measurement region
- **length**: length of the region
- **thickness (for type B)**: thickness in micrometers of the type B region ("horseshoe")
- **depth (for type A)**: depth of the "V" shape from the extremity of the membrane
- **depth (for type D)**: depth of the center of the circle
- **radius (for type D)**: radius of the circle

## How to choose parameters?

We expose only the most important parameters to the user so that it's easier for him to choose them. Some of them like **timestep** and **pixel size** are fixed when recording the video, but others might need some consideration from the user like **step** or **window size**, which depend on how fast the pollen tube grows or how large it is.
Cytoplasm region parameters are entirely up to the user's convenience and can be changed freely, although their default values still ensure good measures. We will discuss the parameters that affect the finding of the tip as finding the right tip is the basis for having good measurements.

### Spline knots

To find the tip of the pollen tube, which defines the measurement area, our model heavily relies on the computation of tangents and normals of the pollen tube.

As tangents and normals are derived from the contour shape, it is crucial to have a denoised shape, which is controlled by the fitting of splines to the contour.

The number of contour points to be considered as knots points, **spline knots**, needs to be carefully picked. Choosing a small number of knots tends to smooth the contour spline but if a small number of knots does not work (because the shape is too noisy or irregular), consider increasing it.

### Step parameter

To locate the tip, we first define a region of interest where we are looking for the tip. This is done by looking at the difference between the current frame and one frame in the future.
We assume that the pollen tube has been growing between those frames and we are now able to locate the region of interest.

The parameter to look for is **step**, which is the number of frames in the future that we use for comparison.

You may want to adapt this parameter according to *how fast* the pollen tube is growing in the video. Picking a small step is suited for fast growing pollen tube videos.

## Output description

Here we will list the outputs of the application (the output directory is by default `temp`, however you can set it to your convenience):

- **video.mp4**: a video showing the tip and measurement region on top of the original video
- **membrane_xs.csv**: an array of size `N * 1` containing the curvilinear abscissa used for membrane measurements (measurements were interpolated to fit on this abscissa so that we can compare measurements along the membrane between frames)
- **membrane_curvs.csv**: an array of size `(n_frames - step) * N` containing the curvatures of the membrane at each point on the curvilinear abscissa
- **membrane_intensities.csv**: an array of size `(n_frames - step) * N` containing the intensities of the membrane at each point on the curvilinear abscissa
- **normals.npy**: an array of array of size `(n_frames - step)` where each array contains the normals vectors of the membrane at each point on the curvilinear abscissa at a frame
- **contours.npy**: an array of array of size `(n_frames - step)` where each array contains the contour of the tube at a frame
- **displacements.npy**: an array of array of size `(n_frames - step)` where each array contains the displacement of the membrane at each point on the curvilinear abscissa at a frame
- **data.csv**: dataframe with `n_frames - step` rows containing the following columns:
  - area_growth: area of the difference between the pollen tube at step `t` and the pollen tube at step `t + 1`
  - cytoplasm_intensity_mean: mean intensity inside the mask defined by region (B or C)
  - tip_size: the size of the tip defined by the distance between its two extremities
  - growth_vec_x,y: the x, y coordinates of the main growth direction at step `t` (computed from `t` to `t + step` with step defined by the parameter `tip.step`)
  - tip_x,y: the x, y coordinates of the tip
  - growth_from_direction: this is the distance between the tip at step `t` and the contour at step `t + step`, following the main growth direction (i.e. we project the tip along the main growth direction towards the next contour)
  - growth_from_tip: the distance between the tip at step `t` and the tip at step `t + step`
  - growth_direction_angle: angle of the growth vector compared to the first growth vector
  - time: the time scale (number of frame or ms)
  ![measures](data/assets/measures.png)
  ![measures_2](data/assets/measures_2.png)

Additionnaly, the GUI provides a few plots for convenience, namely:

- Line plot of plasma membrane mean intensity with respect to time
- Line plot of cytoplasm mean intensity with respect to time
- Line plot of area growth with respect to time
- A heatmap with time as x, curvilinear abscissa as y and intensity as color from the membrane measures (made from the data in **membrane_intensities.csv** wrt **membrane_xs.csv**)
