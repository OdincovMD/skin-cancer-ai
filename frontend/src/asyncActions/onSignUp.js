import { BACKEND_URL } from "../imports/URLS"

import { registerAction } from "../store/userReducer"
import { errorAction } from "../store/userReducer"

export const onSignUp = (registerData) => {
  return async (dispatch) => {
    requestState = {
      userData: {
        name: null,
        surname: nullз
      },
      error: null
    }

    try {
      let response = await fetch(`${BACKEND_URL}/register`, {
        method: "POST",
        headers: {
          "Content-type": "application/json",
          "accept": "application/json"
        },
        body: JSON.stringify(registerData)
      })
  
      if (!response.ok) {
        requestState.error = response.status
        dispatch(errorAction(requestState))
      }
  
      var responseJSON = await response.json()
      if (!responseJSON.isValid) {
        requestState.error = "Логин уже используется"
        dispatch(errorAction(requestState))
      }
  
      requestState.error = null
      requestState.userData.name = registerData.firstName,
      requestState.userData.surname =  registerData.lastName
      dispatch(registerAction(requestState))
    }
  
    catch (err) {
      requestState.error = err
      dispatch(errorAction(requestState))
    }
  }
}