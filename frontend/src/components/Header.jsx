import React from "react"
import { useDispatch, useSelector } from "react-redux"
import { Link, useLocation } from "react-router-dom"
import { LogIn, Menu, Scan, ShieldCheck, User } from "lucide-react"

import { useAvatarObjectUrl } from "../hooks/useAvatarObjectUrl"
import { SIGN_IN, PROFILE } from "../imports/ENDPOINTS"
import { noError } from "../store/userReducer"

const Header = ({ toggleSidebar }) => {
  const dispatch = useDispatch()
  const userInfo = useSelector((state) => state.user)
  const { pathname } = useLocation()

  const firstName = userInfo.userData?.firstName || ""
  const lastName = userInfo.userData?.lastName || ""
  const displayName = `${firstName} ${lastName}`.trim()
  const initials =
    (firstName.charAt(0) || "").toUpperCase() +
    (userInfo.userData?.lastName?.charAt(0) || "").toUpperCase()

  const avatarUrl = useAvatarObjectUrl(
    userInfo.accessToken,
    userInfo.avatarRevision
  )

  return (
    <header className="fixed left-0 right-0 top-0 z-50 h-14 border-b border-slate-200 bg-white/92 shadow-sm backdrop-blur-xl">
      <div className="mx-auto flex h-full max-w-screen-2xl items-center px-3 sm:px-4">
        <button
          onClick={toggleSidebar}
          className="-ml-1 flex h-10 w-10 items-center justify-center rounded-lg border border-transparent text-slate-500 transition-colors hover:border-slate-200 hover:bg-slate-50 hover:text-slate-900"
          aria-label="Меню"
        >
          <Menu size={20} />
        </button>

        <Link to="/" className="ml-3 flex items-center gap-3 select-none">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-med-600 text-white shadow-sm">
            <Scan size={18} strokeWidth={2.2} />
          </div>
          <div className="leading-tight">
            <span className="block text-sm font-bold tracking-tight text-slate-950 sm:text-base">
              Skin Cancer AI
            </span>
            <span className="hidden text-[11px] font-medium text-slate-400 sm:block">
              Дерматоскопическая диагностика
            </span>
          </div>
        </Link>

        <div
          className="ml-auto flex items-center gap-2 sm:gap-3"
          onClick={() => {
            if (pathname !== SIGN_IN) dispatch(noError())
          }}
        >
          <div className="hidden items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700 md:flex">
            <ShieldCheck size={14} />
            ML-сервис активен
          </div>

          {userInfo.userData?.id && userInfo.accessToken ? (
            <Link
              to={PROFILE}
              className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white py-1 pl-1 pr-2 shadow-sm transition-colors hover:bg-slate-50 sm:pr-3"
            >
              {avatarUrl ? (
                <img
                  src={avatarUrl}
                  alt=""
                  className="h-8 w-8 shrink-0 rounded-md object-cover ring-1 ring-slate-200"
                />
              ) : (
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-teal-50 text-xs font-semibold text-med-700 ring-1 ring-teal-100">
                  {initials || <User size={16} />}
                </div>
              )}
              <div className="hidden min-w-0 leading-tight sm:block">
                <span className="block max-w-36 truncate text-sm font-semibold text-slate-800">
                  {displayName || "Профиль"}
                </span>
                <span className="block text-[11px] text-slate-400">
                  Личный кабинет
                </span>
              </div>
            </Link>
          ) : (
            <Link
              to={SIGN_IN}
              className="flex items-center gap-2 rounded-lg bg-med-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-med-700"
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
