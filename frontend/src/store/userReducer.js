const defaultUser = {
    name: "Сосал",
    surname: null,
    error: null
    // email: null,
    // avatar: null,
}

const REGISTER = "REGISTER"
const LOGIN = "LOGIN"
const SIGN_OUT = "SIGN_OUT"
const ERROR = "ERROR"

export const userReducer = (state = defaultUser, action) => {
    switch (action.type) {
      case REGISTER:
        return {...state, name: action.payload.name, surname: action.payload.surname, error: null}
      case LOGIN:
        return {...state, name: action.payload.name, surname: action.payload.surname, error: null}
      case SIGN_OUT:
        return {...state, name: null, surname: null, error: null}
      case ERROR:
        return {...state, name: null, surname: null, error: action.payload.error}
      default:
        return state
    }
  }

export const registerAction = (payload) => ({type: REGISTER, payload: payload})
export const loginAction = (payload) => ({type: LOGIN, payload: payload})
export const signOutAction = (payload) => ({type: SIGN_OUT, payload: payload})
export const errorAction = (payload) => ({type: ERROR, payload: payload})