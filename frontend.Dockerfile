FROM nginx:alpine
COPY kukangsort-ai.html /usr/share/nginx/html/index.html
EXPOSE 80
