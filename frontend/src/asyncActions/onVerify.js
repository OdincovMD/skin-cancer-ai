import { env } from "../imports/ENV"
import { createAsyncThunk } from "@reduxjs/toolkit"

export const onVerify = createAsyncThunk("user/onVerify", async ({data, endpoint}) => {

    let requestState = {
      userData: {
        id: null,
        firstName: null,
        lastName: null,
        email: null,
        // avatar: null,
      },
      error: null
    }

    try {
      let response = await fetch(`${env.BACKEND_URL}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-type": "application/json",
          "accept": "application/json"
        },
        body: JSON.stringify(data)
      })

      if (!response.ok) {
        requestState.error = response.status
        return requestState
      }
      
      var responseJSON = await response.json()
      if (responseJSON.error) {
        requestState.error = responseJSON.error
        return requestState
      }

      // Всё ок
      requestState.userData = responseJSON.userData
      return requestState
    }

    catch (err) {
      requestState.error = `Ошибка обработки запроса: ${err}`
      return requestState
    }
  }
)