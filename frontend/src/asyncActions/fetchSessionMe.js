import { createAsyncThunk } from "@reduxjs/toolkit"

import { env } from "../imports/ENV"
import { ME } from "../imports/ENDPOINTS"
import { bearerAuthHeaders } from "../imports/authHeaders"

export const fetchSessionMe = createAsyncThunk(
  "user/fetchSessionMe",
  async (_, { getState }) => {
    const token = getState().user.accessToken
    if (!token) {
      return { skipped: true }
    }
    const base = env.BACKEND_URL.replace(/\/$/, "")
    const res = await fetch(`${base}${ME}`, {
      method: "GET",
      headers: {
        accept: "application/json",
        ...bearerAuthHeaders(token),
      },
    })
    if (!res.ok) {
      return { error: res.status, userData: null }
    }
    const data = await res.json()
    if (data.error || !data.userData) {
      return { error: data.error || "me_failed", userData: null }
    }
    return { error: null, userData: data.userData }
  }
)
