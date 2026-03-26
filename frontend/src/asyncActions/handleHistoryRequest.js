import { fetchWithAuth } from "./fetchWithAuth"
import { env } from "../imports/ENV"
import { GET_HISTORY } from "../imports/ENDPOINTS"

export const handleHistoryRequest = async (accessToken) => {
  const base = env.BACKEND_URL.replace(/\/$/, "")

  try {
    const response = await fetchWithAuth(accessToken, `${base}${GET_HISTORY}`, {
      method: "POST",
      headers: {
        "Content-type": "application/json",
        "accept": "application/json"
      },
      body: "{}"
    })

    if (!response.ok) {
      let msg = `Произошла ошибка: ${response.status}`
      if (response.status === 403) {
        try {
          const errBody = await response.json()
          if (errBody?.detail) msg = String(errBody.detail)
        } catch {
          /* ignore */
        }
      }
      alert(msg)
      return []
    }

    const responseJSON = await response.json()
    return Array.isArray(responseJSON) ? responseJSON : []
  }
  catch (err) {
    alert(`Ошибка: ${err}`)
    return []
  }
}