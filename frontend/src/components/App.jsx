// components/App.jsx
import React, { useEffect, useState } from "react"
import { useDispatch, useSelector } from "react-redux"
import { Routes, Route, Navigate } from "react-router-dom"

import { onPageReload } from "../store/userReducer.js"

import Header from "./Header.jsx"
import Sidebar from "./Sidebar.jsx"
import Home from "../pages/Home"
import SignIn from "../pages/SignIn"
import SignUp from "../pages/SignUp"
import About from "../pages/About"
import Profile from "../pages/Profile"

import { SIGN_IN, PROFILE } from "../imports/ENDPOINTS.js"


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
          <Routes path="/" element={<Layout />}>
            <Route index element={<Home />}/>
            <Route path="/signin" element={userInfo.userData.id ?  <Navigate to={PROFILE} /> : <SignIn />} />
            <Route path="/signup" element={userInfo.userData.id ?  <Navigate to={PROFILE} /> : <SignUp />} />
            <Route path="/about" element={<About />} />
            <Route path="/profile" element={userInfo.userData.id ? <Profile /> : <Navigate to={SIGN_IN} />}/>
          </Routes>
        </div>
      </main>
    </div>
  )
}

const Layout = () => {
  return (
    <div>
      <Outlet /> {/* Where nested routes will render */}
    </div>
  );
 }

export default App