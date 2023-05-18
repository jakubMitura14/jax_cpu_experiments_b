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