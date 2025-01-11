# ==================================
# Use an official Python runtime as a parent image
# See the uv docs for more details:
#   https://docs.astral.sh/uv/guides/integration/docker/
FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.cargo/bin/:$PATH"FROM python:3.10-slim



# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1


# Sync the project into a new environment, using the frozen lockfile
WORKDIR /app
RUN uv sync --frozen

# Copy the rest of the application
COPY . .

# Presuming there is a `my_app` command provided by the project
CMD ["uv", "run", "my_app"]

# Once the project is installed, you can either activate the project virtual environment by placing its binary directory at the front of the path:
ENV PATH="/app/.venv/bin:$PATH"

# Or you can run the project command directly:
RUN uv run some_script.py