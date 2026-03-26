import React from "react"
import { useDispatch, useSelector } from "react-redux"
import { Link, useNavigate, useLocation } from "react-router-dom"
import { Home, LogIn, UserPlus, Info, LogOut, X } from "lucide-react"

import { HOME, SIGN_IN, SIGN_UP, ABOUT } from "../imports/ENDPOINTS"
import { defaultState, noError } from "../store/userReducer"

const iconMap = {
  "Главная":     Home,
  "Войти":       LogIn,
  "Регистрация": UserPlus,
  "О проекте":   Info,
  "Выйти":       LogOut,
}

const Sidebar = ({ isOpen, onClose }) => {
  const dispatch = useDispatch()
  const userInfo = useSelector((state) => state.user)
  const navigate = useNavigate()
  const { pathname } = useLocation()

  const isAuthed = Boolean(userInfo.userData?.id && userInfo.accessToken)

  const menuItems = [
    { text: "Главная", path: HOME },
    ...(isAuthed
      ? [{ text: "О проекте", path: ABOUT }, { text: "Выйти", path: SIGN_IN }]
      : [
          { text: "Войти", path: SIGN_IN },
          { text: "Регистрация", path: SIGN_UP },
          { text: "О проекте", path: ABOUT },
        ]),
  ]

  const handleItemClick = (item) => {
    if (pathname !== item.path) dispatch(noError())
    if (item.text === "Выйти") {
      dispatch(defaultState())
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
        className={`fixed left-0 top-14 z-40 h-[calc(100vh-3.5rem)] w-64 bg-white border-r border-gray-200 transition-transform duration-200 ease-out
        ${isOpen ? "translate-x-0" : "-translate-x-full"}`}
      >
        <div className="flex h-full flex-col">
          <div className="flex items-center justify-between px-4 py-3 lg:hidden">
            <span className="text-xs font-semibold uppercase tracking-wider text-gray-400">
              Навигация
            </span>
            <button
              onClick={onClose}
              className="rounded-lg p-1.5 text-gray-400 hover:bg-gray-100 hover:text-gray-600 transition-colors"
            >
              <X size={18} />
            </button>
          </div>

          <nav className="flex-1 overflow-y-auto px-3 py-2">
            <ul className="space-y-1">
              {menuItems.map((item) => {
                const Icon = iconMap[item.text] || Home
                const isActive = pathname === item.path && item.text !== "Выйти"
                const isLogout = item.text === "Выйти"

                return (
                  <li key={item.text}>
                    {isLogout ? (
                      <button
                        onClick={() => handleItemClick(item)}
                        className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-red-600 transition-colors hover:bg-red-50"
                      >
                        <Icon size={18} />
                        <span>{item.text}</span>
                      </button>
                    ) : (
                      <Link
                        to={item.path}
                        onClick={() => handleItemClick(item)}
                        className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors
                        ${
                          isActive
                            ? "bg-med-50 text-med-700 font-medium"
                            : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
                        }`}
                      >
                        <Icon size={18} />
                        <span>{item.text}</span>
                      </Link>
                    )}
                  </li>
                )
              })}
            </ul>
          </nav>

          <div className="border-t border-gray-100 px-4 py-3">
            <p className="text-[11px] text-gray-400 leading-relaxed">
              Skin Cancer AI &mdash; диагностика новообразований кожи
            </p>
          </div>
        </div>
      </aside>
    </>
  )
}

export default Sidebar
