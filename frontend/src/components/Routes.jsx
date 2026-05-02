import React from "react"
import { useSelector } from "react-redux"
import { Routes, Route, Navigate } from "react-router-dom"

import {
  API_DOCS,
  FORGOT_PASSWORD,
  HOME,
  WORKSPACE,
  RESET_PASSWORD,
  SIGN_IN,
  PROFILE,
  VERIFY_EMAIL,
  VK_CALLBACK,
  VK_LINK,
} from "../imports/ENDPOINTS.js"

import Home from "../pages/Home.jsx"
import Workspace from "../pages/Workspace.jsx"
import SignIn from "../pages/SignIn.jsx"
import SignUp from "../pages/SignUp.jsx"
import About from "../pages/About.jsx"
import ApiDocs from "../pages/ApiDocs.jsx"
import Profile from "../pages/Profile.jsx"
import VerifyEmail from "../pages/VerifyEmail.jsx"
import ForgotPassword from "../pages/ForgotPassword.jsx"
import ResetPassword from "../pages/ResetPassword.jsx"
import VkAuthCallback from "../pages/VkAuthCallback.jsx"
import VkLinkConfirm from "../pages/VkLinkConfirm.jsx"

const AppRoutes = () => {
  const userInfo = useSelector(state => state.user)
  const isAuthed = Boolean(userInfo.userData?.id && userInfo.accessToken)

  return (
      <Routes>
        <Route index element={<Home />}/>
        <Route path={WORKSPACE} element={<Workspace />}/>
        <Route path="/signin" element={isAuthed ?  <Navigate to={PROFILE} /> : <SignIn />} />
        <Route path="/signup" element={isAuthed ?  <Navigate to={PROFILE} /> : <SignUp />} />
        <Route path="/about" element={<About />} />
        <Route path={API_DOCS} element={<ApiDocs />} />
        <Route path={VERIFY_EMAIL} element={<VerifyEmail />} />
        <Route path={VK_CALLBACK} element={<VkAuthCallback />} />
        <Route path={VK_LINK} element={<VkLinkConfirm />} />
        <Route path={FORGOT_PASSWORD} element={<ForgotPassword />} />
        <Route path={RESET_PASSWORD} element={<ResetPassword />} />
        <Route path="/profile" element={isAuthed ? <Profile /> : <Navigate to={SIGN_IN} />}/>
        <Route path="/*" element={<Navigate to={HOME}/>}/>
      </Routes>
  )
}

export default AppRoutes
