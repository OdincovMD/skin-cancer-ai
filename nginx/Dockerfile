FROM nginx:alpine

RUN apk add --no-cache curl

RUN rm /etc/nginx/conf.d/default.conf
RUN rm /etc/nginx/nginx.conf

COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80