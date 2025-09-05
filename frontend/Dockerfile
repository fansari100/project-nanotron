# syntax=docker/dockerfile:1.7

FROM node:20-bookworm-slim AS builder

WORKDIR /build

ENV CI=1 \
    NODE_OPTIONS=--max-old-space-size=4096

COPY package.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm install --no-audit --no-fund

COPY . .
RUN npm run build

# ---- runtime ----
FROM nginx:1.27-alpine

RUN apk add --no-cache tini && \
    rm -rf /etc/nginx/conf.d/*

COPY --from=builder /build/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
ENTRYPOINT ["/sbin/tini", "--"]
CMD ["nginx", "-g", "daemon off;"]
