# Environment Setup

Every real benchmark task should document one of the following:

- a task-local Docker setup committed alongside the task package
- explicit manual setup instructions that collaborators can execute directly

For a manual setup, this file should normally include:

- runtime version requirements
- package installation commands
- external services or datasets that must be available
- hardware requirements
- a smoke-test command

If a Docker workflow is provided instead, this file should point to the task-local Docker entrypoint and explain the expected invocation.
