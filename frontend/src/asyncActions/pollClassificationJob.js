import { fetchWithAuth } from "./fetchWithAuth"
import { env } from "../imports/ENV"

const jobStorageKey = (userId) => `classification_job_${userId}`

export function savePendingJob(userId, jobId) {
  sessionStorage.setItem(jobStorageKey(userId), String(jobId))
}

export function clearPendingJob(userId) {
  sessionStorage.removeItem(jobStorageKey(userId))
}

export function getPendingJob(userId) {
  return sessionStorage.getItem(jobStorageKey(userId))
}

const emptyClassification = () => ({
  feature_type: null,
  structure: null,
  properties: [],
  final_class: null,
})

/**
 * Опрашивает GET /classification-jobs/{job_id} до completed/error или таймаута.
 * @returns {{ classification: object, imageToken: string|null }}
 */
export async function pollClassificationJob({
  jobId,
  userId,
  accessToken,
  intervalMs = 2000,
  maxAttempts = 600,
}) {
  const base = env.BACKEND_URL.replace(/\/$/, "")
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const r = await fetchWithAuth(
      accessToken,
      `${base}/classification-jobs/${jobId}`
    )
    if (r.status === 404) {
      clearPendingJob(userId)
      throw new Error("Задание не найдено")
    }
    if (!r.ok) {
      throw new Error(`Ошибка запроса статуса: ${r.status}`)
    }
    const data = await r.json()
    const imageToken = data.image_token ?? null
    if (data.status === "completed") {
      clearPendingJob(userId)
      const res = data.result
      const classification =
        res !== null && res !== undefined && res !== ""
          ? res
          : emptyClassification()
      return { classification, imageToken }
    }
    if (data.status === "error") {
      clearPendingJob(userId)
      const err = data.result
      if (err && typeof err === "object" && err.detail !== undefined) {
        const d = err.detail
        return {
          classification: {
            detail: typeof d === "string" ? d : JSON.stringify(d),
          },
          imageToken: null,
        }
      }
      return {
        classification: { detail: "Ошибка классификации" },
        imageToken: null,
      }
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs))
  }
  throw new Error("Превышено время ожидания классификации")
}
