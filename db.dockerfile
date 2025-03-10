# Use the official PostgreSQL image
FROM postgres:latest

# Set environment variables for PostgreSQL
ENV POSTGRES_DB=captcha_db
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=password

# Expose PostgreSQL port
EXPOSE 5432