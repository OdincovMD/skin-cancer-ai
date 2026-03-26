import React from "react"
import { useDispatch, useSelector } from "react-redux"
import { Link, useLocation } from "react-router-dom"
import { Menu, Scan, User, LogIn } from "lucide-react"

import { useAvatarObjectUrl } from "../hooks/useAvatarObjectUrl"
import { SIGN_IN, PROFILE } from "../imports/ENDPOINTS"
import { noError } from "../store/userReducer"

const Header = ({ toggleSidebar }) => {
  const dispatch = useDispatch()
  const userInfo = useSelector((state) => state.user)
  const pathname = useLocation()

  const firstName = userInfo.userData?.firstName || ""
  const initials =
    (firstName.charAt(0) || "").toUpperCase() +
    (userInfo.userData?.lastName?.charAt(0) || "").toUpperCase()

  const avatarUrl = useAvatarObjectUrl(
    userInfo.accessToken,
    userInfo.avatarRevision
  )

  return (
    <header className="fixed top-0 left-0 right-0 h-14 bg-white/95 backdrop-blur border-b border-gray-200 z-50">
      <div className="flex items-center h-full px-4 max-w-screen-2xl mx-auto">
        <button
          onClick={toggleSidebar}
          className="p-2 -ml-1 rounded-lg text-gray-500 hover:bg-gray-100 hover:text-gray-700 transition-colors"
          aria-label="Меню"
        >
          <Menu size={20} />
        </button>

        <Link to="/" className="ml-3 flex items-center gap-2 select-none">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-med-600 text-white">
            <Scan size={18} />
          </div>
          <span className="text-lg font-bold text-gray-900 tracking-tight">
            Skin Cancer&nbsp;<span className="font-normal text-med-600">AI</span>
          </span>
        </Link>

        <div
          className="ml-auto flex items-center gap-3"
          onClick={() => {
            if (pathname !== SIGN_IN) dispatch(noError())
          }}
        >
          {userInfo.userData?.id && userInfo.accessToken ? (
            <Link
              to={PROFILE}
              className="flex items-center gap-2.5 rounded-full py-1 pl-1 pr-3 transition-colors hover:bg-gray-100"
            >
              {avatarUrl ? (
                <img
                  src={avatarUrl}
                  alt=""
                  className="h-8 w-8 shrink-0 rounded-full object-cover ring-2 ring-gray-100"
                />
              ) : (
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-med-100 text-med-700 text-xs font-semibold">
                  {initials || <User size={16} />}
                </div>
              )}
              <span className="text-sm font-medium text-gray-700 hidden sm:inline">
                {firstName}
              </span>
            </Link>
          ) : (
            <Link
              to={SIGN_IN}
              className="flex items-center gap-2 rounded-lg bg-med-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-med-700"
            >
              <LogIn size={16} />
              <span className="hidden sm:inline">Войти</span>
            </Link>
          )}
        </div>
      </div>
    </header>
  )
}

export default Header
