## A. Prerequisites
### 1. Docker Desktop
- If you are using WSL2, ensure Docker Desktop has WSL integration enabled (under Resources/WSL integration/Enable integration with my default WSL distro)
### 2. CUDA and NVIDIA Driver
- Reference: https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
- Common Error: 
    ```
    (HTTP code 500) server error - failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: error running prestart hook #0: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy' nvidia-container-cli: requirement error: unsatisfied condition: cuda>=12.9, please update your driver to a newer version, or use an earlier cuda container: unknown
    ```
- The pre-built vLLM Docker image is built with CUDA 12.9 (or newer).
- Because of that, it enforces a requirement on the host’s GPU driver / CUDA compatibility: on container start, nvidia-container-cli checks that host driver + CUDA support meets cuda>=12.9.
- Two installations as per this link: https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
    1. CUDA Toolkit
    2. Update NVIDIA Driver ("GeForce Game Ready Driver")
- Both the CUDA toolkit and driver have to be installed for CUDA to function: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/#install-the-cuda-software

### Behavior
- The token for huggingface and the model name to deploy can be spcified in the .env file. the template is in the .env.template file.
- Upon the container first start, the model is downloaded to the specified folder on the host and deployed to the container
- At the next container start or recreation, the specified location is used as cache if the same model is specified. Else the new model is downloaded.
- A new container is required if another model is needed.