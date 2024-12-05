# Use Node.js 22.12.3 as the base image
FROM node:22.12.0

# Set the working directory in the container
WORKDIR /frontend

# Copy the package.json and package-lock.json
COPY ./frontend/package.json ./frontend/package-lock.json ./

# Install the React dependencies
RUN npm install

# Copy the rest of the React app's source code
COPY ./frontend /frontend

# Build the React app
RUN npm run build

# Expose port 3000 to the outside world (adjust as needed)
EXPOSE 3000

# Serve the app using a simple static file server
CMD ["npm", "start"]
