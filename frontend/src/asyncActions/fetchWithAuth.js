import store from "../store"
import { defaultState } from "../store/userReducer"

export async function fetchWithAuth(accessToken, url, options = {}) {
  const headers = new Headers(options.headers || {})
  if (accessToken) {
    headers.set("Authorization", `Bearer ${accessToken}`)
  }
  const response = await fetch(url, { ...options, headers })
  if (response.status === 401) {
    store.dispatch(defaultState())
  }
  return response
}
