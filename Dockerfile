# Install uv
FROM python:3.12-slim AS builder
RUN apt-get update
RUN apt-get install git -y
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON=python3.12
ENV UV_PYTHON_DOWNLOADS=never 

# Change the working directory to the `app` directory
WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-editable

RUN --mount=type=secret,id=guard,target=/root/.guardrailsrc \
    /app/.venv/bin/guardrails hub install hub://guardrails/toxic_language

# Copy the project into the intermediate image
ADD ./src/chatbot /app/src/chatbot
ADD ./pyproject.toml /app
ADD ./uv.lock /app

# Sync the project
RUN --mount=type=cache,target=/root/.cache \
    uv sync \
#        --locked \
        --no-dev \
        --no-editable

FROM python:3.12-slim

# Copy the environment, but not the source code
COPY --from=builder  /app /app
RUN mkdir -p /tmp/ch_db
COPY ./ch_db /tmp/ch_db
ENV VIRTUAL_ENV=/app/.venv
ENV PATH=/app/.venv/bin:${PATH}

# Run the application
CMD ["python", "-m", "streamlit", "run", "/app/src/chatbot/app.py"]