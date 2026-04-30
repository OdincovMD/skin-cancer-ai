import React from "react"
import { useDispatch, useSelector } from "react-redux"
import { Link, useNavigate, useLocation } from "react-router-dom"
import {
  BookOpen,
  Home,
  Info,
  KeyRound,
  LogIn,
  LogOut,
  Scan,
  ShieldCheck,
  UserPlus,
  UserRoundCog,
  X,
} from "lucide-react"

import { API_DOCS, HOME, SIGN_IN, SIGN_UP, ABOUT, PROFILE } from "../imports/ENDPOINTS"
import { useAvatarObjectUrl } from "../hooks/useAvatarObjectUrl"
import { notifyOtherTabsSessionCleared } from "../imports/sessionSync"
import { defaultState, noError } from "../store/userReducer"

const iconMap = {
  "Главная": Home,
  "Личный кабинет": UserRoundCog,
  "Войти": LogIn,
  "Регистрация": UserPlus,
  "О проекте": Info,
  "Выйти": LogOut,
  "Документация API": BookOpen,
}

const navigationGroups = {
  main: "Рабочая область",
  account: "Аккаунт",
}

const Sidebar = ({ isOpen, onClose }) => {
  const dispatch = useDispatch()
  const userInfo = useSelector((state) => state.user)
  const navigate = useNavigate()
  const { pathname } = useLocation()

  const isAuthed = Boolean(userInfo.userData?.id && userInfo.accessToken)
  const avatarUrl = useAvatarObjectUrl(
    userInfo.accessToken,
    userInfo.avatarRevision
  )
  const firstName = userInfo.userData?.firstName || ""
  const lastName = userInfo.userData?.lastName || ""
  const displayName = `${firstName} ${lastName}`.trim() || "Пользователь"
  const initials =
    (firstName.charAt(0) || "").toUpperCase() +
    (lastName.charAt(0) || "").toUpperCase()

  const menuItems = [
    { text: "Главная", path: HOME, group: "main" },
    { text: "Документация API", path: API_DOCS, group: "main", badge: "v1" },
    ...(isAuthed
      ? [
          { text: "Личный кабинет", path: PROFILE, group: "account" },
          { text: "О проекте", path: ABOUT, group: "account" },
          { text: "Выйти", path: SIGN_IN, group: "account" },
        ]
      : [
          { text: "Войти", path: SIGN_IN, group: "account" },
          { text: "Регистрация", path: SIGN_UP, group: "account" },
          { text: "О проекте", path: ABOUT, group: "account" },
        ]),
  ]
  const groupedMenuItems = Object.entries(navigationGroups)
    .map(([group, label]) => ({
      group,
      label,
      items: menuItems.filter((item) => item.group === group),
    }))
    .filter((section) => section.items.length > 0)

  const handleItemClick = (item) => {
    if (pathname !== item.path) dispatch(noError())
    if (item.text === "Выйти") {
      dispatch(defaultState())
      notifyOtherTabsSessionCleared()
      navigate(SIGN_IN)
    }
    onClose?.()
  }

  return (
    <>
      {isOpen && (
        <div
          className="fixed inset-0 top-14 z-30 bg-black/20 backdrop-blur-sm lg:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={`fixed left-0 top-14 z-40 h-[calc(100vh-3.5rem)] w-72 border-r border-slate-200 bg-white shadow-xl shadow-slate-950/5 transition-transform duration-200 ease-out lg:w-64
        ${isOpen ? "translate-x-0" : "-translate-x-full"}`}
      >
        <div className="flex h-full flex-col">
          <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3 lg:hidden">
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-med-600 text-white">
                <Scan size={16} />
              </div>
              <span className="text-sm font-semibold text-slate-900">Навигация</span>
            </div>
            <button
              onClick={onClose}
              className="rounded-lg p-1.5 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-700"
              aria-label="Закрыть меню"
            >
              <X size={18} />
            </button>
          </div>
          <nav className="flex-1 overflow-y-auto px-3 py-4">
            <div className="space-y-5">
              {groupedMenuItems.map((section) => (
                <div key={section.group}>
                  <p className="mb-2 px-3 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                    {section.label}
                  </p>
                  <ul className="space-y-1">
                    {section.items.map((item) => {
                      const Icon = iconMap[item.text] || Home
                      const isActive = pathname === item.path && item.text !== "Выйти"
                      const isLogout = item.text === "Выйти"

                      return (
                        <li key={item.text}>
                          {isLogout ? (
                            <button
                              onClick={() => handleItemClick(item)}
                              className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-red-600 transition-colors hover:bg-red-50"
                            >
                              <Icon size={18} />
                              <span>{item.text}</span>
                            </button>
                          ) : (
                            <Link
                              to={item.path}
                              onClick={() => handleItemClick(item)}
                              className={`group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors ${
                                isActive
                                  ? "bg-med-50 font-semibold text-med-700 ring-1 ring-med-100"
                                  : "text-slate-600 hover:bg-slate-100 hover:text-slate-950"
                              }`}
                            >
                              <Icon
                                size={18}
                                className={
                                  isActive
                                    ? "text-med-700"
                                    : "text-slate-400 group-hover:text-med-700"
                                }
                              />
                              <span className="min-w-0 flex-1 truncate">
                                {item.text}
                              </span>
                              {item.badge && (
                                <span
                                  className={`rounded-full px-2 py-0.5 text-[10px] font-bold uppercase ${
                                    isActive
                                      ? "bg-med-100 text-med-700"
                                      : "bg-teal-50 text-med-700"
                                  }`}
                                >
                                  {item.badge}
                                </span>
                              )}
                            </Link>
                          )}
                        </li>
                      )
                    })}
                  </ul>
                </div>
              ))}
            </div>
          </nav>

        </div>
      </aside>
    </>
  )
}

export default Sidebar
