FROM node:18-alpine

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

EXPOSE 3000

ARG BACKEND_URL
RUN echo "export const env = { \
            BACKEND_URL: '${BACKEND_URL}'\
            }"\
          > /app/src/imports/ENV.js

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
