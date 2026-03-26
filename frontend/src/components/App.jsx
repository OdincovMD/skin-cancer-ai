import React, { useState } from "react"

import Header from "./Header.jsx"
import Sidebar from "./Sidebar.jsx"
import AppRoutes from "./Routes.jsx"

const App = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)

  return (
    <div className="min-h-screen bg-gray-50">
      <Header toggleSidebar={() => setIsSidebarOpen((o) => !o)} />
      <Sidebar isOpen={isSidebarOpen} onClose={() => setIsSidebarOpen(false)} />

      <main
        className={`pt-14 transition-all duration-200 ${
          isSidebarOpen ? "lg:pl-64" : "pl-0"
        }`}
      >
        <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6 lg:px-8">
          <AppRoutes />
        </div>
      </main>
    </div>
  )
}

export default App
