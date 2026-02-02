/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow reading files from parent directory
  experimental: {
    serverComponentsExternalPackages: [],
  },
};

module.exports = nextConfig;
