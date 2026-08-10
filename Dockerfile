# Backend-neutral model-stager image:
#
#   docker build -t model-stager:latest .
#
# The default image copies and renames artifacts. ONNX remains ONNX unless
# ONNX_BACKEND=tensorrt is explicitly requested. A conversion-capable variant
# supplies trtexec through the swappable base:
#
#   docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/tensorrt:25.12-py3 \
#     -t model-stager:tensorrt .
#
# A TensorRT engine is specific to the GPU, driver, and TensorRT version that
# built it. Two consequences follow for that optional variant.
#
#   - This container needs a GPU. As a Kubernetes init container it must request
#     nvidia.com/gpu, and it holds the device for the length of the build.
#   - Its TensorRT version must match the server's. NVIDIA Triton Inference Server
#     (https://github.com/triton-inference-server/server) 25.12 and this base both
#     ship TensorRT 10.14; a mismatch produces an engine the server refuses to
#     load, reported as a version error at load time rather than here.
#
# The neutral image needs neither a GPU nor TensorRT.
ARG BASE_IMAGE=busybox:glibc

FROM ${BASE_IMAGE}

COPY bin/model-stager /usr/local/bin/model-stager
RUN chmod +x /usr/local/bin/model-stager

ENV STAGE_DIR=/staging
ENV MODEL_REPOSITORY=/model_repository

# No SERVER_EXEC: this image stages and exits. Starting a server is opt-in, and
# not what an init container should do by default.
ENTRYPOINT ["/usr/local/bin/model-stager"]
