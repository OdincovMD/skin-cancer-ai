export const LS_SESSION_REFRESH_KEY = "skin_session_refresh"

export function notifyOtherTabsSessionMayHaveChanged() {
  try {
    localStorage.setItem(LS_SESSION_REFRESH_KEY, String(Date.now()))
  } catch (_) {
    /* ignore quota / private mode */
  }
}
