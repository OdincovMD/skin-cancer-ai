import React, { useEffect, useState } from "react"
import { useDispatch, useSelector } from "react-redux"
import { Routes, Route, Navigate, useLocation } from "react-router-dom"

import { HOME, SIGN_IN, PROFILE } from "../imports/ENDPOINTS.js"

import Home from "../pages/Home.jsx"
import SignIn from "../pages/SignIn.jsx"
import SignUp from "../pages/SignUp.jsx"
import About from "../pages/About.jsx"
import Profile from "../pages/Profile.jsx"

const AppRoutes = () => {
  const dispatch = useDispatch()
  const userInfo = useSelector(state => state.user)

  const pathname = useLocation()

  return (
      <Routes path="/" element={<Layout />}>
        <Route index element={<Home />}/>
        <Route path="/signin" element={userInfo.userData.id ?  <Navigate to={PROFILE} /> : <SignIn />} />
        <Route path="/signup" element={userInfo.userData.id ?  <Navigate to={PROFILE} /> : <SignUp />} />
        <Route path="/about" element={<About />} />
        <Route path="/profile" element={userInfo.userData.id ? <Profile /> : <Navigate to={SIGN_IN} />}/>
        <Route path="/*" element={<Navigate to={HOME}/>}/>
      </Routes>
  )
}

const Layout = () => {
  return (
    <div>
      <Outlet /> {/* Where nested routes will render */}
    </div>
  )
}
  
export default AppRoutes