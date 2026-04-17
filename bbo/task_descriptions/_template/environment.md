# Environment Setup

Provide one of the following for every task package:

- a task-local Docker setup such as `Dockerfile`, `docker-compose.yml`, or a `docker/` directory
- explicit manual environment setup instructions that allow a collaborator to reproduce the task runtime

Recommended contents for this file when manual setup is used:

- required Python version or system runtime
- package manager and install commands
- external system dependencies
- GPU or accelerator assumptions, if any
- a minimal smoke-test command that confirms the environment is ready
