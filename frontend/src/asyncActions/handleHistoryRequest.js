import { env } from "../imports/ENV"
import { GET_HISTORY } from "../imports/ENDPOINTS"

export const handleHistoryRequest = async (user_id) => {

  const data = { user_id: user_id }

  try {
    let response = await fetch(`${env.BACKEND_URL}${GET_HISTORY}`, {
      method: "POST",
      headers: {
        "Content-type": "application/json",
        "accept": "application/json"
      },
      body: JSON.stringify(data)
    })

    if (!response.ok) {
    alert(`Произошла ошибка: ${response.status}`)
    return
    }

    let responseJSON = await response.json()

    return responseJSON
  }
  catch (err) {
    alert(`Ошибка: ${err}`)
    return {result: null}
  }
}