import { BACKEND_URL } from "../imports/URLS"
import { createAsyncThunk } from "@reduxjs/toolkit"

export const onVerify = createAsyncThunk("user/onVerify", async ({data, endpoint}) => {

    console.log(1)

    let requestState = {
      userData: {
        id: null,
        firstName: null,
        lastName: null,
        email: null,
        // avatar: null,
      },
      error: null,
    }

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
        console.log(response)
        return requestState
      }
      
      var responseJSON = await response.json()
      if (responseJSON.error) {
        requestState.error = `Ошибка данных: ${responseJSON.error}`
        console.log(responseJSON)
        return requestState
      }

      // Всё ок
      requestState.userData = responseJSON.userData
      console.log(responseJSON)
      return requestState
    }

    catch (err) {
      requestState.error = `Ошибка обработки запроса: ${err}`
      console.log(requestState)
      return requestState
    }
  }
)