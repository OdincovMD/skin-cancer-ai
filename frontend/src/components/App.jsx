import React, { useEffect, useState } from "react"
import { useDispatch } from "react-redux"

import { fetchSessionMe } from "../asyncActions/fetchSessionMe"
import { LS_SESSION_REFRESH_KEY } from "../imports/sessionSync"
import Header from "./Header.jsx"
import Sidebar from "./Sidebar.jsx"
import AppRoutes from "./Routes.jsx"

const App = () => {
  const dispatch = useDispatch()
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)

  useEffect(() => {
    const onStorage = (e) => {
      if (e.key === LS_SESSION_REFRESH_KEY && e.newValue != null) {
        dispatch(fetchSessionMe())
      }
    }
    window.addEventListener("storage", onStorage)
    return () => window.removeEventListener("storage", onStorage)
  }, [dispatch])

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
