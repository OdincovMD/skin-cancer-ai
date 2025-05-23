// components/Header.jsx
import React from "react"
import { useDispatch, useSelector  } from "react-redux"
import { Link, useLocation } from "react-router-dom"

import { SIGN_IN, PROFILE } from "../imports/ENDPOINTS"
import { defaultState, noError } from "../store/userReducer"

const Header = ({ toggleSidebar }) => {

  const dispatch = useDispatch()
  const userInfo = useSelector(state => state.user)

  const pathname = useLocation()

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
          className="ml-auto flex flex-row items-center space-x-4" 
          onClick={() => {
            (pathname != SIGN_IN) &&
            dispatch(noError()) 
          }}
        >
          { userInfo.userData.id ? (
              <Link
                to={PROFILE}
                className="text-gray-600"
              >
                <div className="flex flex-row items-center">
                  <p className="inline-block mr-2 text-gray-900 hover:text-gray-500">
                    {`${userInfo.userData.firstName}`}
                  </p>
                  <div className="w-8 h-8 rounded-full overflow-hidden inline-block">
                    <img
                      src={userInfo.userData.id == 1 ? "/images/PP.png" : "/images/image.png"}
                      alt="Profile Picture"
                      className="w-full h-full object-cover"
                    />
                  </div>
                </div>
              </Link>
            ) : (
              <Link
                to={SIGN_IN}
                className="text-gray-900 hover:text-gray-500"
              >
                <span className="block truncate">Войдите, чтобы получить доступ к системе</span>
              </Link>
          )}
        </div>
      </div>
    </header>
  )
}

export default Header