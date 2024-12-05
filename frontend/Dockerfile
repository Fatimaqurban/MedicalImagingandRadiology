# Use Node.js 22.12.3 as the base image
FROM node:22.12.0

# Set the working directory inside the container
WORKDIR /app/frontend

# Copy the package.json and package-lock.json files into the container
COPY frontend/package.json frontend/package-lock.json ./

# Install the React dependencies
RUN npm install

# Copy the rest of the React app's source code into the container
COPY frontend/ /app/frontend/

# Expose port 3000 for the app to be accessible outside
EXPOSE 3000

# Run the React app
CMD ["npm", "start"]
