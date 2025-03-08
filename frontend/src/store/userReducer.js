const defaultUser = {
    userData: {
      id: null,
      firstName: null,
      lastName: null,
      email: null,
      // avatar: null,
    },
    error: null,
}

const VERIFY = "VERIFY"
const SIGN_OUT = "SIGN_OUT"

export const userReducer = (state = defaultUser, action) => {
    switch (action.type) {
      case VERIFY:
        return {...action.payload}
      case SIGN_OUT:
        return defaultUser
      default:
        return state
    }
  }

export const verifyAction = (payload) => ({type: VERIFY, payload: payload})
export const signOutAction = (payload) => ({type: SIGN_OUT, payload: payload})