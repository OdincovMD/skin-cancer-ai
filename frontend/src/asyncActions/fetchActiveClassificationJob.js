import { env } from "../imports/ENV"
import { CLASSIFICATION_JOBS_ACTIVE } from "../imports/ENDPOINTS"

/**
 * Активное задание пользователя (pending/processing) из БД.
 * @returns {Promise<{ job_id: number, status: string, file_name: string } | null>}
 */
export async function fetchActiveClassificationJob(userId) {
  const base = env.BACKEND_URL.replace(/\/$/, "")
  const r = await fetch(
    `${base}${CLASSIFICATION_JOBS_ACTIVE}?user_id=${encodeURIComponent(userId)}`
  )
  if (r.status === 204) return null
  if (!r.ok) {
    throw new Error(`Не удалось получить активное задание: ${r.status}`)
  }
  return r.json()
}
