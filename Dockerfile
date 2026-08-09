# model-stager image.
#
#   docker build -t model-stager:trt .
#
# The default base carries trtexec, because compiling ONNX to a TensorRT engine
# is the one staging step that cannot be done anywhere else: an engine is
# specific to the GPU, driver, and TensorRT version that built it. Two
# consequences follow.
#
#   - This container needs a GPU. As a Kubernetes init container it must request
#     nvidia.com/gpu, and it holds the device for the length of the build.
#   - Its TensorRT version must match the server's. Triton 25.12 and this base
#     both ship TensorRT 10.14; a mismatch produces an engine the server refuses
#     to load, reported as a version error at load time rather than here.
#
# A repository with no TensorRT in it needs neither. Staging is then only copying
# and renaming, and a far smaller base does it:
#
#   docker build --build-arg BASE_IMAGE=busybox:glibc -t model-stager:slim .
ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt:25.12-py3

FROM ${BASE_IMAGE}

COPY bin/model-stager /usr/local/bin/model-stager
RUN chmod +x /usr/local/bin/model-stager

ENV STAGE_DIR=/staging
ENV MODEL_REPOSITORY=/models/repo

# No SERVER_EXEC: this image stages and exits. Starting a server is opt-in, and
# not what an init container should do by default.
ENTRYPOINT ["/usr/local/bin/model-stager"]
