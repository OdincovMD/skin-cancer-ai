import { fetchWithAuth } from "./fetchWithAuth"
import { env } from "../imports/ENV"
import { UPLOAD_FILE } from "../imports/ENDPOINTS"
import {
  pollClassificationJob,
  savePendingJob,
  clearPendingJob,
} from "./pollClassificationJob"

export const handleUploadImage = async ({ id, fileData, accessToken }) => {
  const formData = new FormData()
  formData.append("file", fileData)

  const base = env.BACKEND_URL.replace(/\/$/, "")

  try {
    const response = await fetchWithAuth(accessToken, `${base}${UPLOAD_FILE}`, {
      method: "POST",
      body: formData,
    })

    if (response.status === 429) {
      const defaultMsg =
        "Слишком много запросов. Дождитесь завершения текущей классификации."
      let errMsg = defaultMsg
      try {
        const body = await response.json()
        if (body?.detail != null) {
          errMsg =
            typeof body.detail === "string"
              ? body.detail
              : JSON.stringify(body.detail)
        }
      } catch {
        /* тело не JSON — оставляем defaultMsg */
      }
      alert(errMsg)
      return {
        feature_type: null,
        structure: null,
        properties: [],
        final_class: null,
      }
    }

    if (!response.ok) {
      let msg = `Произошла ошибка: ${response.status}`
      if (response.status === 403) {
        try {
          const body = await response.json()
          if (body?.detail) {
            msg =
              typeof body.detail === "string"
                ? body.detail
                : JSON.stringify(body.detail)
          }
        } catch {
          /* ignore */
        }
      }
      alert(msg)
      return {
        feature_type: null,
        structure: null,
        properties: [],
        final_class: null,
      }
    }

    const data = await response.json()

    if (data.job_id != null) {
      savePendingJob(id, data.job_id)
      try {
        return await pollClassificationJob({
          jobId: data.job_id,
          userId: id,
          accessToken,
        })
      } catch (e) {
        clearPendingJob(id)
        alert(String(e.message || e))
        return {
          feature_type: null,
          structure: null,
          properties: [],
          final_class: null,
        }
      }
    }

    return data
  } catch (err) {
    alert(`Ошибка: ${err}`)
    return {
      feature_type: null,
      structure: null,
      properties: [],
      final_class: null,
    }
  }
}
