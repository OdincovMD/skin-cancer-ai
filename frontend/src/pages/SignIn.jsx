import React, { useState } from "react"
import { useDispatch, useSelector } from "react-redux"
import { Link, Navigate } from "react-router-dom"
import { Eye, EyeOff } from "lucide-react"

import { onVerify } from "../asyncActions/onVerify"
import { HOME, SIGN_IN, SIGN_UP } from "../imports/ENDPOINTS"

const SignIn = () => {

  const dispatch = useDispatch()
  const userInfo = useSelector(state => state.user)

  const [isPasswordVisible, setIsPasswordVisible] = useState(false)

  const [login, setLogin] = useState(null)
  const [password, setPassword] = useState(null)

  const togglePasswordVisibility = () => {
    setIsPasswordVisible((prevState) => !prevState);
  }

  const handleSubmit = (event) => {
    event.preventDefault()
    dispatch(onVerify({data: {login: login, password: password}, endpoint: SIGN_IN }))
    setLogin(null)
    setPassword(null)
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-100 p-6">
      <div className="w-full max-w-md rounded-2xl bg-white p-8 shadow-lg">
        <h2 className="mb-6 text-center text-2xl font-semibold text-gray-700">Вход в систему</h2>
        <form className="space-y-4" onSubmit={handleSubmit}>

          <input
            type="text"
            placeholder="Логин"
            className="w-full rounded-lg border border-gray-300 p-3 outline-blue-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            onChange={(ans) => { setLogin(ans.target.value) }}
          />

          <div
            className="flex flex-row w-full rounded-lg border border-gray-300 p-3 focus-within:border-blue-500 focus-within:ring-1 focus-within:ring-blue-500"
          >
            <input
              type={isPasswordVisible ? "text" : "password"}
              placeholder="Пароль"
              className="w-full border-none focus:outline-none"
              onChange={(ans) => { setPassword(ans.target.value) }}
            />
            <button 
              type="button"
              title={isPasswordVisible ? "Скрыть пароль" : "Показать пароль"}
              className="cursor-pointer text-gray-400 rounded-e-md focus:outline-none focus-visible:text-blue-500 hover:text-blue-500 transition-colors" onClick={togglePasswordVisibility} aria-label={isPasswordVisible ? "Hide password" : "Show password"} aria-pressed={isPasswordVisible} aria-controls="password" >
              {isPasswordVisible ? ( <EyeOff size={20} aria-hidden="true" /> ) : ( <Eye size={20} aria-hidden="true" /> )}
            </button>
          </div>

          <div className="text-red-500">
            {userInfo.error}
          </div>

          <button
            type="submit"
            className="w-full rounded-lg bg-blue-600 px-4 py-3 text-white font-semibold transition hover:bg-blue-700"
          >
            Войти
          </button>

          <div className="flex flex-row items-center justify-center">
            <span className="block truncate white">Еще не зарегестрированы?</span>
            <Link
              to={SIGN_UP}
              className="text-blue-600"
            >
              <span className="underline ml-1 transition hover:text-blue-900">{`Регистрация`}</span>
            </Link>
          </div>

          <div>
            {userInfo.userData.id ? <Navigate to={HOME}/> : null}
          </div>
          
        </form>
      </div>
    </div>
  )
}

export default SignIn