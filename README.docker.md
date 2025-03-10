# Captcha Solver Demo Webapp

![webapp](/media/webapp.png)

Deploy a local webapp to test out the models with Docker.

You'll need to install docker on your local desktop first.

## Model Weights

You can download my model weights [here](https://drive.google.com/file/d/19E32XYX-TuEMgJQHBd3BQjauPuou7vIY/view?usp=drive_link).

Unzip the downloaded weights at project root under `checkpoints`.

Or use your own weights, rename your checkpoints as `checkpoint.pth` leave them in the corresponding folder (e.g. `checkpoints/train_vit/checkpoint.pth` for ViT's weight).

## Deploy Demo with Docker Compose (Recommended)

```
docker compose up -d
```

You can now access the webapp on http://localhost:8000, and the backend sits on http://localhost:8090.

## Deploy Demo with Individual Docker Containers

Three docker images are needed to run the webapp: database, backend, and frontend.

They can all be built together using docker compose as specified in the `docker-compose.yml` file, 
see the above section for how to build with docker compose (the recommended way).

### Building the database

Build the database container:

```
docker build -t captcha-db -f db.dockerfile .
```

Run the database container:

```
docker run -p 5432:5432 -it --rm captcha-db:latest
```

### Building the backend

Build the backend container:

```
docker build -t captcha-backend -f backend.dockerfile .
```

Run the backend container:

```
docker run -p 8090:8000 -it --rm -e DB_HOST=localhost DB_PORT=5432 captcha-backend:latest
```

Backend will then be mapped to local host port 8090.

### Building the frontend

Build the frontend container:

```
docker build -t captcha-app --build-arg BACKEND_URL=http://localhost:8090 -f backend.dockerfile .
```

Run the backend container:

```
docker run -p 3000:3000 -it --rm -e DB_HOST=localhost -e DB_PORT=5432 captcha-app:latest
```

You can now access the webapp on http://localhost:3000.