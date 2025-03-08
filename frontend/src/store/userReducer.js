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

const userSlice = createSlice({
  name: "user",
  initialState: defaultUser,
  reducers: {
    defaultState: (state) => { 
      return defaultUser
    }
  },
  extraReducers: (builder) => {
    builder
      .addCase(onVerify.fulfilled, (state, action) => {
        state.userData = action.payload.userData
        state.error = action.payload.error
      })
  }
})

export const { defaultState } = userSlice.actions
export const userReducer = userSlice.reducer