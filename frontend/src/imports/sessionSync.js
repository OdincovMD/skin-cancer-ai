export const LS_SESSION_REFRESH_KEY = "skin_session_refresh"
export const LS_SESSION_SHARE_REQUEST_KEY = "skin_session_share_request"
export const LS_SESSION_SHARE_RESPONSE_KEY = "skin_session_share_response"
export const LS_SESSION_CLEAR_KEY = "skin_session_clear"

function broadcastLocalStorageEvent(key, value) {
  try {
    localStorage.setItem(key, value)
  } catch (_) {
    /* ignore quota / private mode */
  }
}

export function notifyOtherTabsSessionMayHaveChanged() {
  broadcastLocalStorageEvent(LS_SESSION_REFRESH_KEY, String(Date.now()))
}

export function requestSessionFromOtherTabs() {
  broadcastLocalStorageEvent(LS_SESSION_SHARE_REQUEST_KEY, String(Date.now()))
}

export function publishSessionToOtherTabs(userInfoRaw) {
  if (!userInfoRaw) return
  broadcastLocalStorageEvent(
    LS_SESSION_SHARE_RESPONSE_KEY,
    JSON.stringify({
      at: Date.now(),
      userInfo: userInfoRaw,
    })
  )
}

export function publishStoredSessionToOtherTabs() {
  const raw =
    sessionStorage.getItem("userInfo") ?? localStorage.getItem("userInfo")
  publishSessionToOtherTabs(raw)
  notifyOtherTabsSessionMayHaveChanged()
}

export function notifyOtherTabsSessionCleared() {
  broadcastLocalStorageEvent(LS_SESSION_CLEAR_KEY, String(Date.now()))
}
