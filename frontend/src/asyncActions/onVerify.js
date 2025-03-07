import { BACKEND_URL } from "../imports/URLS"

import { verifyAction } from "../store/userReducer"

export const onVerify = async (data, endpoint) => {
  return async (dispatch) => {
    requestState = {
      userData: {
        id: null,
        firstName: null,
        lastName: null,
        // email: null,
        // avatar: null,
      },
      error: null,
    }

    requestHandler: { 
      try {
        let response = await fetch(`${BACKEND_URL}${endpoint}`, {
          method: "POST",
          headers: {
            "Content-type": "application/json",
            "accept": "application/json"
          },
          body: JSON.stringify(data)
        })

        if (!response.ok) {
          requestState.error = `Ошибка бэкенда: ${response.status}`
          break requestHandler
        }

        var responseJSON = await response.json()
        if (responseJSON.error) {
          requestState.error = `Ошибка данных: ${responseJSON.error}`
          break requestHandler
        }

        // Всё ок
        requestState.userData = responseJSON.userData
        break requestHandler
      }

      catch (err) {
        requestState.error = `Ошибка обработки запроса: ${err}`
        break requestHandler
      }
    }

    dispatch(verifyAction(requestState))
  }
}