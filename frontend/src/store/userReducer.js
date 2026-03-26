import { createSlice } from "@reduxjs/toolkit"

import { fetchSessionMe } from "../asyncActions/fetchSessionMe"
import { onVerify } from "../asyncActions/onVerify"

const defaultUser = {
    userData: {
      id: null,
      firstName: null,
      lastName: null,
      email: null,
    },
    accessToken: null,
    emailVerified: true,
    error: null,
    isRememberMeChecked: false,
    verificationResendUntilMs: null,
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
        if (parsedData.userData?.id && !parsedData.accessToken) {
          storage.removeItem('userInfo')
          return defaultUser
        }
        state.userData = parsedData.userData
        state.accessToken = parsedData.accessToken ?? null
        state.error = null
        state.isRememberMeChecked = parsedData.isRememberMeChecked
        state.emailVerified =
          parsedData.emailVerified !== undefined
            ? Boolean(parsedData.emailVerified)
            : true

        let until = parsedData.verificationResendUntilMs
        if (until != null && Number(until) <= Date.now()) {
          until = null
        }
        state.verificationResendUntilMs =
          until != null ? Number(until) : null

        storage.setItem('userInfo', JSON.stringify({
          userData: state.userData,
          accessToken: state.accessToken,
          error: state.error,
          isRememberMeChecked: state.isRememberMeChecked,
          emailVerified: state.emailVerified,
          verificationResendUntilMs: state.verificationResendUntilMs,
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
    },
    mergeUserData: (state, action) => {
      const p = action.payload || {}
      if (p.firstName !== undefined) {
        state.userData.firstName = p.firstName
      }
      if (p.lastName !== undefined) {
        state.userData.lastName = p.lastName
      }
      const storage = state.isRememberMeChecked ? localStorage : sessionStorage
      if (state.accessToken) {
        storage.setItem('userInfo', JSON.stringify({
          userData: state.userData,
          accessToken: state.accessToken,
          error: state.error,
          isRememberMeChecked: state.isRememberMeChecked,
          emailVerified: state.emailVerified,
          verificationResendUntilMs: state.verificationResendUntilMs,
        }))
      }
    },
    setVerificationResendCooldownFromSeconds: (state, action) => {
      const sec = Math.max(0, Number(action.payload) || 0)
      state.verificationResendUntilMs =
        sec > 0 ? Date.now() + sec * 1000 : null
      const storage = state.isRememberMeChecked ? localStorage : sessionStorage
      if (state.accessToken) {
        storage.setItem('userInfo', JSON.stringify({
          userData: state.userData,
          accessToken: state.accessToken,
          error: state.error,
          isRememberMeChecked: state.isRememberMeChecked,
          emailVerified: state.emailVerified,
          verificationResendUntilMs: state.verificationResendUntilMs,
        }))
      }
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(onVerify.pending, (state) => {
        state.error = null
      })
      .addCase(onVerify.fulfilled, (state, action) => {
        const prevUntil = state.verificationResendUntilMs
        state.userData = action.payload.userData
        state.error = action.payload.error
        state.accessToken = action.payload.accessToken ?? null
        state.emailVerified =
          action.payload.emailVerified !== undefined
            ? Boolean(action.payload.emailVerified)
            : true

        const storage = state.isRememberMeChecked ? localStorage : sessionStorage

        if (state.accessToken) {
          const secOk = state.emailVerified
            ? 0
            : (Number(action.payload.verificationResendAfterSeconds) || 0)
          state.verificationResendUntilMs =
            secOk > 0 ? Date.now() + secOk * 1000 : null
        } else {
          state.verificationResendUntilMs = prevUntil
        }

        if (state.accessToken) {
          storage.setItem('userInfo', JSON.stringify({
            userData: state.userData,
            accessToken: state.accessToken,
            error: state.error,
            isRememberMeChecked: state.isRememberMeChecked,
            emailVerified: state.emailVerified,
            verificationResendUntilMs: state.verificationResendUntilMs,
          }))
        }
      })
      .addCase(fetchSessionMe.fulfilled, (state, action) => {
        if (action.payload?.skipped) return
        if (action.payload?.error || !action.payload?.userData) return
        const ud = action.payload.userData
        state.userData = {
          id: ud.id ?? null,
          firstName: ud.firstName ?? null,
          lastName: ud.lastName ?? null,
          email: ud.email ?? null,
        }
        state.emailVerified = ud.email_verified === true
        const secMe = state.emailVerified
          ? 0
          : (Number(ud.verification_resend_after_seconds) || 0)
        state.verificationResendUntilMs =
          secMe > 0 ? Date.now() + secMe * 1000 : null
        const storage = state.isRememberMeChecked ? localStorage : sessionStorage
        if (state.accessToken) {
          storage.setItem('userInfo', JSON.stringify({
            userData: state.userData,
            accessToken: state.accessToken,
            error: state.error,
            isRememberMeChecked: state.isRememberMeChecked,
            emailVerified: state.emailVerified,
            verificationResendUntilMs: state.verificationResendUntilMs,
          }))
        }
      })
  }
})

export const {
  defaultState,
  onPageReload,
  toggleRememberMe,
  noError,
  mergeUserData,
  setVerificationResendCooldownFromSeconds,
} = userSlice.actions
export const userReducer = userSlice.reducer