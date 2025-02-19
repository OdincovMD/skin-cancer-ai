FROM node:18-alpine

WORKDIR /app

# !!! Важно: убедитесь, что вы скопировали файлы package.json и package-lock.json
COPY package*.json ./

RUN npm install && \
    npm install -D tailwindcss postcss autoprefixer && \
    npm install lucide-react && \
    npx tailwindcss init -p && \
    npm install easy-magnify-waft

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]