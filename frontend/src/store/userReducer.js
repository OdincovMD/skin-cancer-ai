const defaultState = {
    name: null,
    surname: null,
    email: null,
    avatar: null,
}

const REGISTER = "REGISTER"
const LOGIN = "LOGIN"
const SIGN_OUT = "SIGN_OUT"

export const userReducer = (state = defaultState, action) => {
    switch (action.type) {
      case REGISTER:
  
      case LOGIN:
  
      case SIGN_OUT:
  
      default:
        return state
    }
  }

export const registerAction = (payload) => ({type: REGISTER, payload: payload})
export const loginAction = (payload) => ({type: LOGIN, payload: payload})
export const signOutAction = (payload) => ({type: SIGN_OUT, payload: payload})