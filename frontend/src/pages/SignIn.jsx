import React from "react"
import { useDispatch } from "react-redux"
import { Link } from "react-router-dom"

import { onVerify } from "../asyncActions/onVerify"
import { SIGN_IN, SIGN_UP } from "../imports/ENDPOINTS"

const SignIn = () => {

  const dispatch = useDispatch()

  var [login, password] = [null, null]

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-100 p-6">
      <div className="w-full max-w-md rounded-2xl bg-white p-8 shadow-lg">
        <h2 className="mb-6 text-center text-2xl font-semibold text-gray-700">Вход в систему</h2>
        <form className="space-y-4">

          <input
            type="text"
            placeholder="Логин"
            className="w-full rounded-lg border border-gray-300 p-3 outline-blue-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            onChange={(ans) => { login = ans.target.value }}
          />

          <input
            type="password"
            placeholder="Пароль"
            className="w-full rounded-lg border border-gray-300 p-3 outline-blue-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            onChange={(ans) => { password = ans.target.value }}
          />

          <button
            type="submit"
            className="w-full rounded-lg bg-blue-600 px-4 py-3 text-white font-semibold transition hover:bg-blue-700"
            onClick={() => {
              dispatch(onVerify({login: login, password: password}, SIGN_IN))
              [login, password] = null, null
            }}
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
          
        </form>
      </div>
    </div>
  )
}

export default SignIn