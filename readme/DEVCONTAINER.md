# Dev Container

## Why we use a dev container

This project uses a VS Code Dev Container to provide a consistent, reproducible development environment for everyone working on the forensic linguist agent. A dev container lets VS Code open the project inside a containerized environment with a defined runtime, tools, and editor extensions, so setup differences between machines do not affect development. VS Code uses the `devcontainer.json` file to describe how to create and configure that environment. 

For this project, that is useful because the agent stack depends on a specific Python version, Python tooling, Git, and environment-variable handling for the private LLM backend. Putting that into the dev container avoids “works on my machine” issues and makes onboarding much faster. Dev containers are especially useful when a codebase needs separate tools, libraries, or runtimes that should not depend on the host system.

---

## Requirements

Before starting the dev container, make sure you have:

- [Visual Studio Code](https://code.visualstudio.com/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- The [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

VS Code documents Docker as the standard way to run Dev Containers locally. 

---

## How to start the dev container

### macOS

1. Install Docker Desktop and start it.
2. Install VS Code.
3. Install the Dev Containers extension.
4. Open this project folder in VS Code.
5. Open the Command Palette with `Cmd+Shift+P`.
6. Run `Dev Containers: Rebuild and Reopen in Container` if the configuration already exists, or `Dev Containers: Open Folder in Container...` if starting from a folder workflow.
7. Wait for the container build to finish. VS Code will connect to the container automatically. 

### Windows

1. Install Docker Desktop and start it.
2. Install VS Code.
3. Install the Dev Containers extension.
4. Open this project folder in VS Code.
5. Open the Command Palette with `Ctrl+Shift+P`.
6. Run `Dev Containers: Rebuild and Reopen in Container` if the configuration already exists, or `Dev Containers: Open Folder in Container...` if starting from a folder workflow.
7. Wait for the container build to finish. VS Code will connect to the container automatically. 

### After the first build

The first container build takes longer because the image, features, and Python packages must be installed. Later opens are faster because VS Code reuses the built container when possible. 

---

## Environment variables

This project uses:

```env
WEBIS_KEY=your_api_key_here
OPENWEBUI_WEBIS_KEY=your_api_key_here
````
