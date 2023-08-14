# See here for image contents: https://hub.docker.com/r/jupyter/datascience-notebook/

FROM jupyter/datascience-notebook

# We want to run common-debian.sh from here:
# https://github.com/microsoft/vscode-dev-containers/tree/main/script-library#development-container-scripts
# But that script assumes that the main non-root user (in this case jovyan)
# is in a group with the same name (in this case jovyan).  So we must first make that so.
COPY library-scripts/common-debian.sh /tmp/library-scripts/
USER root
RUN apt-get update \
 && groupadd jovyan \
 && usermod -g jovyan -a -G users jovyan \
 && bash /tmp/library-scripts/common-debian.sh \
 && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
# COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install pip install "jax[cpu]" 



RUN pip3 install numpy \
        h5py \
        SimpleITK \
        flax \
        toolz \
        matplotlib \
        more-itertools  \
        ml-collections \
        Pillow \
        scikit-image \
        imageio

RUN pip3 install einops \
        optax \
        ml_collections
RUN pip3 --disable-pip-version-check --no-cache-dir install pip install flax

RUN pip3 --disable-pip-version-check --no-cache-dir  install jraph
RUN pip3 --disable-pip-version-check --no-cache-dir  install numpyro[cpu]
RUN pip3 --disable-pip-version-check --no-cache-dir  install  seaborn
RUN pip3 --disable-pip-version-check --no-cache-dir  install  opencv-python
RUN pip3 --disable-pip-version-check --no-cache-dir  install  ott-jax
# RUN pip3 --disable-pip-version-check --no-cache-dir  install  orbax #jax_smi
RUN pip3 --disable-pip-version-check --no-cache-dir  install  tensorflow
RUN pip3 --disable-pip-version-check --no-cache-dir  install  zenodo_get==1.3.4 xnat
RUN pip3 --disable-pip-version-check --no-cache-dir  install  pydicom medpy

# RUN set -ex \
#     && apt-get -qq update \
#     && apt-get -qq install -y --no-install-recommends \
#         build-essential cmake \
#         wget unzip \
#         libhdf5-103-1 libhdf5-dev \
#         libopenblas0 libopenblas-dev \
#         libprotobuf23 libprotobuf-dev \
#         libjpeg62-turbo libjpeg62-turbo-dev \
#         libpng16-16 libpng-dev \
#         libtiff5 libtiff-dev \
#         libwebp6 libwebp-dev \
#         libopenjp2-7 libopenjp2-7-dev \
#         libtbb2 libtbb-dev \
#         libeigen3-dev \
#         tesseract-ocr tesseract-ocr-por libtesseract-dev \
#         python3 python3-pip python3-numpy python3-dev \
#     && wget -q --no-check-certificate https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -O opencv.zip \
#     && wget -q --no-check-certificate https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip -O opencv_contrib.zip \
#     && unzip -qq opencv.zip -d /opt && rm -rf opencv.zip \
#     && unzip -qq opencv_contrib.zip -d /opt && rm -rf opencv_contrib.zip \
#     && cmake \
#         -D CMAKE_BUILD_TYPE=RELEASE \
#         -D CMAKE_INSTALL_PREFIX=/usr/local \
#         -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
#         -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
#         -D OPENCV_ENABLE_NONFREE=ON \
#         -D WITH_JPEG=ON \
#         -D WITH_PNG=ON \
#         -D WITH_TIFF=ON \
#         -D WITH_WEBP=ON \
#         -D WITH_JASPER=ON \
#         -D WITH_EIGEN=ON \
#         -D WITH_TBB=ON \
#         -D WITH_LAPACK=ON \
#         -D WITH_PROTOBUF=ON \
#         -D WITH_V4L=OFF \
#         -D WITH_GSTREAMER=OFF \
#         -D WITH_GTK=OFF \
#         -D WITH_QT=OFF \
#         -D WITH_CUDA=OFF \
#         -D WITH_VTK=OFF \
#         -D WITH_OPENEXR=OFF \
#         -D WITH_FFMPEG=OFF \
#         -D WITH_OPENCL=OFF \
#         -D WITH_OPENNI=OFF \
#         -D WITH_XINE=OFF \
#         -D WITH_GDAL=OFF \
#         -D WITH_IPP=OFF \
#         -D BUILD_OPENCV_PYTHON3=ON \
#         -D BUILD_OPENCV_PYTHON2=OFF \
#         -D BUILD_OPENCV_JAVA=OFF \
#         -D BUILD_TESTS=OFF \
#         -D BUILD_IPP_IW=OFF \
#         -D BUILD_PERF_TESTS=OFF \
#         -D BUILD_EXAMPLES=OFF \
#         -D BUILD_ANDROID_EXAMPLES=OFF \
#         -D BUILD_DOCS=OFF \
#         -D BUILD_ITT=OFF \
#         -D INSTALL_PYTHON_EXAMPLES=OFF \
#         -D INSTALL_C_EXAMPLES=OFF \
#         -D INSTALL_TESTS=OFF \
#         /opt/opencv-${OPENCV_VERSION} \
#     && make -j$(nproc) \
#     && ln -s /usr/local/lib/python3.9/site-packages/cv2 /usr/local/lib/python3.9/dist-packages/cv2 \
#     && make install \
#     && rm -rf /opt/build/* \
#     && rm -rf /opt/opencv-${OPENCV_VERSION} \
#     && rm -rf /opt/opencv_contrib-${OPENCV_VERSION} \
#     && apt-get -qq remove -y \
#         software-properties-common \
#         build-essential cmake \
#         libhdf5-dev \
#         libprotobuf-dev \
#         libjpeg62-turbo-dev \
#         libpng-dev \
#         libtiff-dev \
#         libwebp-dev \
#         libopenjp2-7-dev \
#         libtbb-dev \
#         libtesseract-dev \
#         python3-dev \
#     && apt-get -qq autoremove \
#     && apt-get -qq clean

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

USER jovyan




    # numba==0.56.4+1.g772622d0d\
    # # cupy \
    # black==23.1.0\
    # pandas==1.5.2 \
    # numpy==1.22.2 \
    # comet-ml==3.32.1 \
    # h5py==3.6.0 \
    # SimpleITK==2.2.1\
    # batchgenerators==0.24 \
    # dask==2022.11.1 \
    # nibabel==5.0.0\
    # # intensity-normalization[ants] \
    # warp-lang \
    # report_guided_annotation==0.2.7\
    # ott-jax \ 
    # evojax[extra] \
    # jraph \
    # flax==0.6.7 \
    # toolz==0.12.0\
    # ipyparallel==8.4.1\
    # optuna==3.1.0\
    # # equinox \
    # einops==0.6.0\
    # torchio==0.18.87 \
    # jax-triton==0.1.2 \
    # opencv-python==4.7.0.68 \
    # dm-pix==0.3.4 \
    # visu3d==1.4.0 \
    # ml-collections==0.1.1 \
    # #https://stackoverflow.com/questions/63938552/how-to-run-tensorboard-in-vscode
    # tensorboard-plugin-profile \
    # jax-smi==1.0.2 \
    # tensorly==0.8.0 \
    # tqdm==4.64.1 \
    # kymatio==0.3.0 \
    # jaxwt \
    # matplotlib \
    # PyWavelets==1.4.1 \
    # Pillow==9.4.0 \
    # scikit-image==0.20.0 \
    # imageio==2.26.0 \
    # jax-cosmo==0.1.0 \
    # more-itertools 