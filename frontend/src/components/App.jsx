import React, { useEffect, useState } from "react"
import { useDispatch, useSelector } from "react-redux"

import { fetchSessionMe } from "../asyncActions/fetchSessionMe"
import { LS_SESSION_REFRESH_KEY } from "../imports/sessionSync"
import { defaultState } from "../store/userReducer"
import Header from "./Header.jsx"
import Sidebar from "./Sidebar.jsx"
import AppRoutes from "./Routes.jsx"

const App = () => {
  const dispatch = useDispatch()
  const accessTokenExpiresAt = useSelector((state) => state.user.accessTokenExpiresAt)
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)

  useEffect(() => {
    const isExpired = () => {
      if (!accessTokenExpiresAt) return false
      const expiresAtMs = Date.parse(String(accessTokenExpiresAt))
      return Number.isFinite(expiresAtMs) && expiresAtMs <= Date.now()
    }

    const refreshSession = () => {
      if (isExpired()) {
        dispatch(defaultState())
        return
      }
      dispatch(fetchSessionMe())
    }

    refreshSession()

    const onStorage = (e) => {
      if (e.key === LS_SESSION_REFRESH_KEY && e.newValue != null) {
        refreshSession()
      }
    }
    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        refreshSession()
      }
    }
    let expiryTimerId = null
    if (accessTokenExpiresAt) {
      const expiresAtMs = Date.parse(String(accessTokenExpiresAt))
      if (Number.isFinite(expiresAtMs)) {
        const delay = expiresAtMs - Date.now()
        expiryTimerId = window.setTimeout(() => {
          dispatch(defaultState())
        }, Math.max(0, delay))
      }
    }

    window.addEventListener("storage", onStorage)
    window.addEventListener("focus", refreshSession)
    document.addEventListener("visibilitychange", onVisibilityChange)
    return () => {
      if (expiryTimerId != null) window.clearTimeout(expiryTimerId)
      window.removeEventListener("storage", onStorage)
      window.removeEventListener("focus", refreshSession)
      document.removeEventListener("visibilitychange", onVisibilityChange)
    }
  }, [accessTokenExpiresAt, dispatch])

  return (
    <div className="flex min-h-screen flex-col bg-gray-50">
      <Header toggleSidebar={() => setIsSidebarOpen((o) => !o)} />
      <Sidebar isOpen={isSidebarOpen} onClose={() => setIsSidebarOpen(false)} />

      <main
        className={`flex-1 pt-14 transition-all duration-200 ${
          isSidebarOpen ? "lg:pl-64" : "pl-0"
        }`}
      >
        <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6 lg:px-8">
          <AppRoutes />
        </div>
      </main>

      <footer
        className={`border-t border-slate-200 py-6 text-center text-sm text-slate-500 transition-all duration-200 ${
          isSidebarOpen ? "lg:pl-64" : "pl-0"
        }`}
      >
        <a
          href="https://skin-cancer-ai.ru"
          target="_blank"
          rel="noreferrer"
          className="font-medium text-med-600 hover:text-med-700 hover:underline"
        >
          Skin Cancer AI
        </a>{" "}
        &copy; 2026
      </footer>
    </div>
  )
}

export default App
