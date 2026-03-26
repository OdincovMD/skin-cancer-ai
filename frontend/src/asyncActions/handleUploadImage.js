import { fetchWithAuth } from "./fetchWithAuth"
import { env } from "../imports/ENV"
import { UPLOAD_FILE } from "../imports/ENDPOINTS"
import {
  pollClassificationJob,
  savePendingJob,
  clearPendingJob,
} from "./pollClassificationJob"

const emptyClassification = () => ({
  feature_type: null,
  structure: null,
  properties: [],
  final_class: null,
})

function detailFromBody(body) {
  if (body == null || typeof body !== "object") return null
  const d = body.detail
  if (d == null) return null
  if (typeof d === "string") return d
  if (Array.isArray(d) && d[0]?.msg) return String(d[0].msg)
  return JSON.stringify(d)
}

async function parseResponseJson(response) {
  try {
    const text = await response.text()
    if (!text.trim()) return null
    return JSON.parse(text)
  } catch {
    return null
  }
}

export const handleUploadImage = async ({ id, fileData, accessToken }) => {
  const formData = new FormData()
  formData.append("file", fileData)

  const base = env.BACKEND_URL.replace(/\/$/, "")
  const fail = (message) => ({
    error: message,
    classification: emptyClassification(),
    imageToken: null,
  })

  try {
    const response = await fetchWithAuth(accessToken, `${base}${UPLOAD_FILE}`, {
      method: "POST",
      body: formData,
    })

    const body = await parseResponseJson(response)

    if (response.status === 429) {
      return fail(
        detailFromBody(body) ||
          "Слишком много запросов. Дождитесь завершения текущей классификации."
      )
    }

    if (!response.ok) {
      return fail(
        detailFromBody(body) ||
          `Не удалось загрузить файл (${response.status}). Повторите попытку.`
      )
    }

    if (body == null || typeof body !== "object") {
      return fail("Некорректный ответ сервера при загрузке.")
    }

    const data = body

    if (data.job_id != null) {
      savePendingJob(id, data.job_id)
      try {
        const polled = await pollClassificationJob({
          jobId: data.job_id,
          userId: id,
          accessToken,
        })
        return {
          error: null,
          classification: polled.classification,
          imageToken: polled.imageToken,
        }
      } catch (e) {
        clearPendingJob(id)
        return fail(String(e?.message || e))
      }
    }

    return {
      error: null,
      classification: data?.classification ?? data ?? emptyClassification(),
      imageToken: data?.image_token ?? null,
    }
  } catch (err) {
    return fail(String(err?.message || err))
  }
}
