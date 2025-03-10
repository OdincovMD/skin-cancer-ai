// components/Sidebar.jsx
import React from "react"
import { useDispatch, useSelector  } from "react-redux"
import { Link, Navigate } from "react-router-dom"

import { HOME, SIGN_IN, SIGN_UP, ABOUT } from "../imports/ENDPOINTS"
import { defaultState, noError } from "../store/userReducer"

const Sidebar = ({ isOpen }) => {

  const dispatch = useDispatch()
  const userInfo = useSelector(state => state.user)

  const pathname = window.location.pathname

  const menuItemsLogged = [
    { text: "О нас", path: ABOUT },
    { text: "Выйти", path: SIGN_IN}
  ]

  const menuItemsNotLogged = [
    { text: "Войти", path: SIGN_IN },
    { text: "Регистрация", path: SIGN_UP },
    { text: "О нас", path: ABOUT },
  ]

  const menuItemsShow = userInfo.userData.id ?
    menuItemsLogged :
    menuItemsNotLogged

  const menuItems = [
    { text: "Главная", path: HOME },
    ...menuItemsShow,
    ,
  ]

  return (
    <aside
      className={`fixed left-0 top-16 h-[calc(100vh-4rem)] bg-white shadow-lg transition-transform duration-300 
      ${isOpen ? "translate-x-0 w-64" : "-translate-x-full w-64"}`}
    >
      <div className="h-full overflow-hidden">
        <nav className="p-4">
          <ul className="space-y-2">
            {menuItems.map((item, index) => (
              <li 
                key={index} 
                onClick={() => {
                  if (pathname != item.path) { 
                    dispatch(noError())
                  }
                  if (item.text == "Выйти") {
                    dispatch(defaultState())
                    return <Navigate to={SIGN_IN}/>
                  }
                }}>
                <Link
                  to={item.path}
                  className="block w-full p-3 rounded-lg hover:bg-gray-100 text-gray-700 transition-colors"
                >
                  <span className="block truncate">{item.text}</span>
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </aside>
  )
}

export default Sidebar