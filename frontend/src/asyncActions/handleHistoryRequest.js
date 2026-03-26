import { env } from "../imports/ENV"
import { GET_HISTORY } from "../imports/ENDPOINTS"

export const handleHistoryRequest = async (user_id) => {

  const data = { user_id: user_id }
  const base = env.BACKEND_URL.replace(/\/$/, "")

  try {
    const response = await fetch(`${base}${GET_HISTORY}`, {
      method: "POST",
      headers: {
        "Content-type": "application/json",
        "accept": "application/json"
      },
      body: JSON.stringify(data)
    })

    if (!response.ok) {
      alert(`Произошла ошибка: ${response.status}`)
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