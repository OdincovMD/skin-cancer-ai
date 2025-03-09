// components/Header.jsx
import React, { useDebugValue } from "react"
import { Link } from "react-router-dom"
import { useDispatch, useSelector } from "react-redux"

import { SIGN_IN, PROFILE } from "../imports/ENDPOINTS"
import { defaultState } from "../store/userReducer"

const Header = ({ toggleSidebar }) => {

  const dispatch = useDispatch()
  const userInfo = useSelector(state => state.user)

  const pathname = window.location.pathname

  return (
    <header className="fixed top-0 left-0 right-0 h-16 bg-white shadow-md z-50">
      <div className="flex items-center h-full px-4">
        <button
          onClick={toggleSidebar}
          className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
          aria-label="Toggle menu"
        >
          <span className="text-xl">☰</span>
        </button>
        <h1 className="ml-4 text-xl font-bold text-gray-800">Skin</h1>

        <div 
          className="ml-auto flex items-center space-x-4" 
          onClick={() => {
            pathname != SIGN_IN ? 
            dispatch(defaultState()) : 
            null
          }}
        >
          { userInfo.userData.id ? (
              <Link
                to={PROFILE}
                className="text-gray-600"
              >
                <span className="block truncate">{`${userInfo.userData.firstName} ${userInfo.userData.lastName}`}</span>
              </Link>
            ) : (
              <Link
                to={SIGN_IN}
                className="text-gray-600"
              >
                <span className="block truncate">{`Войдите, чтобы получить доступ к системе`}</span>
              </Link>
          )}
        </div>
      </div>
    </header>
  )
}

export default Header