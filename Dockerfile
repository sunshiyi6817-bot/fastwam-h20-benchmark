FROM nvcr.io/nvidia/pytorch:25.09-py3@sha256:f3b4a33fd60d5e0358287bb44d57e91eda45e8f1681fef1bfd6c969371417130
ENV VIRTUAL_ENV=/opt/fastwam-env
ENV PATH="/opt/fastwam-env/bin:${PATH}"
COPY environment/requirements.txt environment/constraints-native.txt /tmp/fastwam-deps/
RUN python -m venv --system-site-packages /opt/fastwam-env     && PIP_CONSTRAINT=/tmp/fastwam-deps/constraints-native.txt DS_BUILD_OPS=0 python -m pip install --no-build-isolation -r /tmp/fastwam-deps/requirements.txt     && python -m pip check     && python -c "import torch; assert torch.__version__ == '2.9.0a0+50eac811a6.nv25.09', torch.__version__"
WORKDIR /aligned/FastWAM
CMD ["bash"]
