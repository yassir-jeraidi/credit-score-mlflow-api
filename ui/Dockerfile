## Multi-stage Next.js (pnpm) Dockerfile

FROM node:22-alpine AS base

WORKDIR /app

ENV NODE_ENV=production

# -----------------------------------------------------------------------------
# 1) Dependencies layer (pnpm install)
# -----------------------------------------------------------------------------
FROM base AS deps

# Enable pnpm via corepack
RUN corepack enable && corepack prepare pnpm@latest --activate

# Copy only dependency manifests for caching
COPY package.json pnpm-lock.yaml ./

# Install dependencies (frozen lockfile for reproducibility)
RUN pnpm install --frozen-lockfile

# -----------------------------------------------------------------------------
# 2) Builder layer (Next.js build with standalone output)
# -----------------------------------------------------------------------------
FROM base AS builder

WORKDIR /app

# Enable pnpm in builder stage
RUN corepack enable && corepack prepare pnpm@latest --activate

# Reuse node_modules from deps stage
COPY --from=deps /app/node_modules ./node_modules

# Copy full source
COPY . .

# Build-time environment (can be overridden by build args)
ARG API_BASE_URL=http://api:8000
ARG NEXT_PUBLIC_API_URL=http://localhost:8000

ENV API_BASE_URL=${API_BASE_URL}
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

# Build the Next.js app (standalone output configured in next.config.ts)
RUN pnpm build

# -----------------------------------------------------------------------------
# 3) Runtime layer (minimal image)
# -----------------------------------------------------------------------------
FROM node:22-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production

# Create non-root user
RUN addgroup --system --gid 1001 nodejs \
  && adduser --system --uid 1001 nextjs

# Copy built assets from builder image
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# Ensure correct ownership
RUN chown -R nextjs:nodejs /app

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
