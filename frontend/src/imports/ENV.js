/** Базовый URL API (в Docker перезаписывается в Dockerfile). */
export const env = {
  BACKEND_URL: import.meta.env.VITE_BACKEND_URL || "/backend",
}
