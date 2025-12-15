import path from "path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable standalone output for Docker
  output: "standalone",

  // Ensure webpack can resolve "@/..." aliases (for Docker/CI builds)
  webpack(config) {
    config.resolve = config.resolve || {};
    config.resolve.alias = {
      ...(config.resolve.alias || {}),
      "@": path.resolve(__dirname, "src"),
    };
    return config;
  },

  // Proxy API requests to FastAPI backend during development
  async rewrites() {
    return [
      {
        source: "/backend/:path*",
        destination: `${process.env.API_BASE_URL || "http://localhost:8000"}/:path*`,
      },
    ];
  },
  // Environment variables available on the client
  env: {
    NEXT_PUBLIC_API_URL: process.env.API_BASE_URL || "http://localhost:8000",
  },
};

export default nextConfig;
