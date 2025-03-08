import { createSlice } from "@reduxjs/toolkit"

import { onVerify } from "../asyncActions/onVerify"

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

const userSlice = createSlice({
  name: "user",
  initialState: defaultUser,
  reducers: {},
  extraReducers: (builder) => {
    builder
      // .addCase(onVerify.loading, (state) => {
      //   state = action.payload
      // })
      .addCase(onVerify.fulfilled, (state, action) => {
        state.userData = action.payload.userData;
        state.error = action.payload.error;
      })
      // .addCase(onVerify.fulfilled, (state, action) => {
      //   state = action.payload
      // })
  }
})

export const userReducer = userSlice.reducer

// export const userReducer = (state = defaultUser, action) => {
//     switch (action.type) {
//       case VERIFY:
//         return {...action.payload}
//       case SIGN_OUT:
//         return defaultUser
//       default:
//         return state
//     }
//   }

// export const verifyAction = (payload) => ({type: VERIFY, payload: payload})
// export const signOutAction = (payload) => ({type: SIGN_OUT, payload: payload})