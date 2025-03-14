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
    isRememberMeChecked: false
}

const userSlice = createSlice({
  name: "user",
  initialState: defaultUser,
  reducers: {
    defaultState: (state) => {
      sessionStorage.removeItem('userInfo')
      localStorage.removeItem('userInfo')
      return defaultUser
    },
    onPageReload: (state) => {
      const userInfoSession = sessionStorage.getItem('userInfo')

      const storage = userInfoSession ? sessionStorage : localStorage
      const userInfo = storage.getItem('userInfo')

      if (userInfo) {
        const parsedData = JSON.parse(userInfo)
        state.userData = parsedData.userData
        state.error = null
        state.isRememberMeChecked = parsedData.isRememberMeChecked

        storage.setItem('userInfo', JSON.stringify({
          userData: state.userData,
          error: state.error,
          isRememberMeChecked: state.isRememberMeChecked
        }))
      }
    },
    toggleRememberMe: (state) => {
      const oldStorage = state.isRememberMeChecked ? localStorage : sessionStorage
      const newStorage = state.isRememberMeChecked ? sessionStorage : localStorage

      const userInfo = oldStorage.getItem('userInfo')
      if (userInfo) {
        newStorage.setItem('userInfo', userInfo)
        oldStorage.removeItem('userInfo')
      }

      state.isRememberMeChecked = !state.isRememberMeChecked
    },
    noError: (state) => {
      return {...state, error: null}
    }
  },
  extraReducers: (builder) => {
    builder
      .addCase(onVerify.pending, (state) => {
        state.error = null
      })
      .addCase(onVerify.fulfilled, (state, action) => {
        state.userData = action.payload.userData
        state.error = action.payload.error

        const storage = state.isRememberMeChecked ? localStorage : sessionStorage

        storage.setItem('userInfo', JSON.stringify({
          userData: state.userData,
          error: state.error,
          isRememberMeChecked: state.isRememberMeChecked,
        }))
      })
  }
})

export const { defaultState, onPageReload, toggleRememberMe, noError } = userSlice.actions
export const userReducer = userSlice.reducer