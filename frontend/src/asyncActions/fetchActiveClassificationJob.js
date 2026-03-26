import { fetchWithAuth } from "./fetchWithAuth"
import { env } from "../imports/ENV"
import { CLASSIFICATION_JOBS_ACTIVE } from "../imports/ENDPOINTS"

export async function fetchActiveClassificationJob(accessToken) {
  const base = env.BACKEND_URL.replace(/\/$/, "")
  const r = await fetchWithAuth(accessToken, `${base}${CLASSIFICATION_JOBS_ACTIVE}`)
  if (r.status === 204) return null
  if (!r.ok) {
    throw new Error(`Не удалось получить активное задание: ${r.status}`)
  }
  return r.json()
}
