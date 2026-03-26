import React from "react"
import { useSelector } from "react-redux"
import { Routes, Route, Navigate } from "react-router-dom"

import { HOME, SIGN_IN, PROFILE } from "../imports/ENDPOINTS.js"

import Home from "../pages/Home.jsx"
import SignIn from "../pages/SignIn.jsx"
import SignUp from "../pages/SignUp.jsx"
import About from "../pages/About.jsx"
import Profile from "../pages/Profile.jsx"
import VerifyEmail from "../pages/VerifyEmail.jsx"

const AppRoutes = () => {
  const userInfo = useSelector(state => state.user)
  const isAuthed = Boolean(userInfo.userData?.id && userInfo.accessToken)

  return (
      <Routes>
        <Route index element={<Home />}/>
        <Route path="/signin" element={isAuthed ?  <Navigate to={PROFILE} /> : <SignIn />} />
        <Route path="/signup" element={isAuthed ?  <Navigate to={PROFILE} /> : <SignUp />} />
        <Route path="/about" element={<About />} />
        <Route path="/verify-email" element={<VerifyEmail />} />
        <Route path="/profile" element={isAuthed ? <Profile /> : <Navigate to={SIGN_IN} />}/>
        <Route path="/*" element={<Navigate to={HOME}/>}/>
      </Routes>
  )
}

export default AppRoutes