FROM node:18-alpine

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

ARG BACKEND_URL
ARG VK_ID_APP_ID
ARG VK_ID_REDIRECT_URI
ARG VK_ID_SCOPES
RUN echo "export const env = { \
            BACKEND_URL: '${BACKEND_URL}',\
            VK_ID_APP_ID: '${VK_ID_APP_ID}',\
            VK_ID_REDIRECT_URI: '${VK_ID_REDIRECT_URI}',\
            VK_ID_SCOPES: '${VK_ID_SCOPES}'\
            }"\
          > /app/src/imports/ENV.js

RUN npm run build && npm install -g serve

EXPOSE 3000

CMD ["serve", "-s", "dist", "-l", "3000"]
