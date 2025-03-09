import { BACKEND_URL } from "../imports/URLS"
import { GET_HISTORY } from "../imports/ENDPOINTS"

export const handleHistoryRequest = async (user_id) => {

  try {
    let response = await fetch(`${BACKEND_URL}${GET_HISTORY}`, {
      method: "POST",
      headers: {
        "Content-type": "application/json",
        "accept": "application/json"
      },
      body: JSON.stringify(user_id)
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