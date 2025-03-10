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
    isLoading: false
}

const userSlice = createSlice({
  name: "user",
  initialState: defaultUser,
  reducers: {
    defaultState: (state) => { 
      return defaultUser
    },
    noError: (state) => { 
      return {...state, error: null}
    }
  },
  extraReducers: (builder) => {
    builder
      .addCase(onVerify.pending, (state) => {
        state.error = null
        state.isLoading = true
      })
      .addCase(onVerify.fulfilled, (state, action) => {
        state.userData = action.payload.userData
        state.error = action.payload.error
        state.isLoading = false
      })
  }
})

export const { defaultState, noError } = userSlice.actions
export const userReducer = userSlice.reducer