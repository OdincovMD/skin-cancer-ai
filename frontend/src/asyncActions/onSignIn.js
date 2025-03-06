import { BACKEND_URL } from "../imports/URLS"

import { loginAction } from "../store/userReducer"
import { errorAction } from "../store/userReducer"

export const onSignIn = async (loginData) => {
  return async (dispatch) => {
    requestState = {
      userData: {
        name: null,
        surname: null
      },
      error: null
    }

    try {
      let response = await fetch(`${BACKEND_URL}/login`, {
        method: "POST",
        headers: {
          "Content-type": "application/json",
          "accept": "application/json"
        },
        body: JSON.stringify(loginData)
      })

      if (!response.ok) {
        requestState.error = response.status
        dispatch(errorAction(requestState))
      }

      var responseJSON = await response.json()
      if (!responseJSON.isValid) {
        requestState.error = "Неверный логин или пароль"
        dispatch(errorAction(requestState))
      }

      requestState.error = null
      requestState.userData.name = responseJSON.name,
      requestState.userData.surname =  responseJSON.surname
      dispatch(loginAction(requestState))
    }

    catch (err) {
      requestState.error = err
      dispatch(errorAction(requestState))
    }
  }
}