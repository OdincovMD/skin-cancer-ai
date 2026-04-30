FROM node:18-alpine

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

ARG BACKEND_URL
RUN echo "export const env = { \
            BACKEND_URL: '${BACKEND_URL}'\
            }"\
          > /app/src/imports/ENV.js

RUN npm run build && npm install -g serve

EXPOSE 3000

CMD ["serve", "-s", "dist", "-l", "3000"]
