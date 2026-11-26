FROM node:22-alpine AS dependencies
WORKDIR /repo/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --legacy-peer-deps

FROM node:22-alpine AS build
WORKDIR /repo/frontend
COPY --from=dependencies /repo/frontend/node_modules ./node_modules
COPY frontend/ ./
COPY backend/openapi.json /repo/backend/openapi.json
COPY shared/languages.json /repo/shared/languages.json
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
RUN npm run build

FROM node:22-alpine AS runtime
ENV NODE_ENV=production
WORKDIR /repo/frontend
COPY --from=build /repo/frontend ./
EXPOSE 3000
CMD ["npm", "run", "start"]
