# Use Node.js 22.12.0 as the base image
FROM node:22.12.0

# Set the working directory in the container
WORKDIR /app/frontend

# Copy the package.json and package-lock.json
COPY frontend/package.json frontend/package-lock.json ./

# Install the React dependencies
RUN npm install

# Copy the rest of the React app's source code from the host into the container
COPY ./frontend /app/frontend/

# Expose port 3000 to the outside world (adjust as needed)
EXPOSE 3000

# Serve the app using npm
CMD ["npm", "start"]
