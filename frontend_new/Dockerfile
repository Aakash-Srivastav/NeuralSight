FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package.json and install dependencies
COPY package*.json ./
RUN npm install

# Copy rest of the frontend code
COPY . .

# Expose port
EXPOSE 3000

# Start dev server
CMD ["npm", "start"]


# # Stage 1: Build React app
# FROM node:18-alpine as build

# # Set working directory
# WORKDIR /app

# # Install dependencies
# COPY package*.json ./
# RUN npm install

# # Copy source files and build
# COPY . .
# RUN npm run build

# # Stage 2: Serve with Nginx
# FROM nginx:alpine

# # Copy built files to Nginx default public folder
# COPY --from=build /app/build /usr/share/nginx/html

# # Copy custom nginx config if needed (optional)
# # COPY nginx.conf /etc/nginx/conf.d/default.conf

# # Expose port (Render detects this)
# EXPOSE 80

# # Start Nginx
# CMD ["nginx", "-g", "daemon off;"]
