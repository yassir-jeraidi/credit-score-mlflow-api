# syntax=docker.io/docker/dockerfile:1
# =============================================================================
# Next.js UI - Multi-Stage Build (Optimized)
# =============================================================================

# =============================================================================
# Stage 0: Base
# =============================================================================
FROM node:22-alpine AS base

# =============================================================================
# Stage 1: Builder - Install dependencies and build
# =============================================================================
FROM base AS builder

WORKDIR /app

# Install dependencies based on the preferred package manager
# Note: Build context is set to project root, so paths are relative to root
COPY ui/package.json ui/yarn.lock* ui/package-lock.json* ui/pnpm-lock.yaml* ui/.npmrc* ./
COPY ui/postcss.config.mjs* ui/tailwind.config.ts* ui/next.config.ts* ./

# Omit --production flag for TypeScript devDependencies
RUN \
  if [ -f yarn.lock ]; then yarn --frozen-lockfile; \
  elif [ -f package-lock.json ]; then npm ci; \
  elif [ -f pnpm-lock.yaml ]; then corepack enable pnpm && pnpm i; \
  else echo "Warning: Lockfile not found." && yarn install; \
  fi

# Copy application source from the ui directory
COPY ui/ .

# Environment variables must be present at build time
ARG API_BASE_URL
ENV API_BASE_URL=${API_BASE_URL}
ARG NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

# Disable telemetry at build time
ENV NEXT_TELEMETRY_DISABLED=1

# Build Next.js based on the preferred package manager
RUN \
  if [ -f yarn.lock ]; then yarn build; \
  elif [ -f package-lock.json ]; then npm run build; \
  elif [ -f pnpm-lock.yaml ]; then pnpm build; \
  else npm run build; \
  fi

# =============================================================================
# Stage 2: Production - Minimal runtime image
# =============================================================================
FROM base AS runner

# Set labels for image metadata
LABEL maintainer="Credit Score MLOps Team" \
      description="Next.js UI - Optimized Multi-Stage Build" \
      version="1.0.0"

WORKDIR /app

# Don't run production as root
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

USER nextjs

# Copy only necessary files from builder
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Environment variables at runtime
ARG API_BASE_URL
ENV API_BASE_URL=${API_BASE_URL}
ARG NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

# Disable telemetry at runtime
ENV NEXT_TELEMETRY_DISABLED=1

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD wget -q --spider http://localhost:3000 || exit 1

CMD ["node", "server.js"]