// components/App.jsx
import React, { useEffect, useState } from "react"
import { useDispatch, useSelector } from "react-redux"

import { onPageReload } from "../store/userReducer.js"

import Header from "./Header.jsx"
import Sidebar from "./Sidebar.jsx"

import AppRoutes from "./Routes.jsx"

const App = () => {

  const dispatch = useDispatch()
  const userInfo = useSelector(state => state.user)

  const [isSidebarOpen, setIsSidebarOpen] = useState(true)

  useEffect(() => {
    dispatch(onPageReload()); // Восстанавливаем состояние при загрузке приложения
  }, [dispatch]);

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen)
  }

  return (
    <div className=" bg-gray-50">
      <Header toggleSidebar={toggleSidebar} />
      <Sidebar isOpen={isSidebarOpen} />
      <main
        className={`pt-16  transition-all duration-300 
        ${isSidebarOpen ? "pl-64" : "pl-0"}`}
      >
        <div className="p-6">
          <AppRoutes />
        </div>
      </main>
    </div>
  )
}

export default App