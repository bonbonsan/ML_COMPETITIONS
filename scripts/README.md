# scripts

Helper shell scripts that streamline Docker and Paperspace workflows.

## GPU Workflow

- `gpu_setup_and_build.sh` – One-time setup on a fresh Paperspace machine (drivers + Docker build).
- `gpu_pull_and_rebuild.sh` – Pull the latest repo and rebuild the Docker image when dependencies change.
- `docker-run_with_port.sh` – Launch the container with GPU access and forward the required ports.
- `start_paperspace.sh` – Create an SSH tunnel from the local machine to the remote Paperspace instance.
- `run_jupyter-lab.sh` – Start JupyterLab inside the container.

## Data Sync

- `pull_data_from_container.sh` – Copy experiment artifacts from the remote container to the local host.
- `push_data_to_container.sh` – Upload local datasets or checkpoints to the remote container.

All scripts assume the repository root as the working directory. Set `PAPERSPACE_PUBLIC_IP` in `.paperspace.env` before running remote commands.
