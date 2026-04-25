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

const isDescriptionTerminal = (status) =>
  status == null || status === "completed" || status === "error"

function descriptionPayload(data) {
  return {
    descriptionStatus: data?.description_status ?? null,
    description: data?.description ?? null,
    descriptionError: data?.description_error ?? null,
    importantLabels: Array.isArray(data?.important_labels)
      ? data.important_labels
      : [],
  }
}

export async function pollClassificationJob({
  jobId,
  userId,
  accessToken,
  intervalMs = 2000,
  maxAttempts = 600,
  onUpdate,
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
    const descriptionState = descriptionPayload(data)
    if (data.status === "completed") {
      const res = data.result
      const classification =
        res !== null && res !== undefined && res !== ""
          ? res
          : emptyClassification()
      const payload = {
        classification,
        imageToken,
        ...descriptionState,
      }
      onUpdate?.(payload)
      if (isDescriptionTerminal(descriptionState.descriptionStatus)) {
        clearPendingJob(userId)
        return payload
      }
    }
    if (data.status === "error") {
      clearPendingJob(userId)
      const err = data.result
      if (err && typeof err === "object" && err.detail !== undefined) {
        const d = err.detail
        const payload = {
          classification: {
            detail: typeof d === "string" ? d : JSON.stringify(d),
          },
          imageToken: null,
          ...descriptionState,
        }
        onUpdate?.(payload)
        return payload
      }
      const payload = {
        classification: { detail: "Ошибка классификации" },
        imageToken: null,
        ...descriptionState,
      }
      onUpdate?.(payload)
      return payload
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs))
  }
  throw new Error("Превышено время ожидания классификации")
}
