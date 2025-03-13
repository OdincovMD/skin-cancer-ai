import React, { useEffect, useState } from "react"
import { useDispatch, useSelector } from "react-redux"
import { Link, useNavigate } from "react-router-dom"
import { Eye, EyeOff } from "lucide-react"

import { onVerify } from "../asyncActions/onVerify"
import { HOME, SIGN_IN, SIGN_UP } from "../imports/ENDPOINTS"
import { mappingInfo } from "../imports/HELPERS"
import { noError, toggleRememberMe } from "../store/userReducer"

const SignIn = () => {

  const defaultFormState = {
    login: "",
    password: "",
  }

  const [isRequestPending, setIsRequestPending] = useState(false)
  const [formState, setFormState] = useState(defaultFormState)

  const dispatch = useDispatch()
  const userInfo = useSelector(state => state.user)
  const navigate = useNavigate()

  const [isPasswordVisible, setIsPasswordVisible] = useState(false)

  const togglePasswordVisibility = () => {
    setIsPasswordVisible((prevState) => !prevState)
  }

  useEffect(() => {
    if (!isRequestPending && userInfo.error) {
      dispatch(noError())
    }
  }, [formState])

  const handleSubmit = async (event) => {
    event.preventDefault()

    setIsRequestPending(true)
    dispatch(onVerify({ 
      data: {
        [mappingInfo.LOGIN]: formState.login,
        [mappingInfo.PASSWORD]: formState.password
      },
      endpoint: SIGN_IN
    }))
    setIsRequestPending(false)

    userInfo.userData.id && navigate(HOME)
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-100 p-6">
      <div className="w-full max-w-md rounded-2xl bg-white p-8 shadow-lg">
        <h2 className="mb-6 text-center text-2xl font-semibold text-gray-700">Вход в систему</h2>
        <form 
          className="flex flex-col justify-center items-center space-y-4"
          onSubmit={handleSubmit}
        >

          <input
            type="text"
            placeholder="Логин"
            className="w-full rounded-lg border border-gray-300 p-3 outline-blue-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            onChange={(ans) => { setFormState({...formState, login: ans.target.value}) }}
          />

          <div
            className="flex flex-row w-full rounded-lg border border-gray-300 p-3 focus-within:border-blue-500 focus-within:ring-1 focus-within:ring-blue-500"
          >
            <input
              name="password"
              type={isPasswordVisible ? "text" : "password"}
              placeholder="Пароль"
              className="w-full border-none focus:outline-none"
              onChange={(ans) => { setFormState({...formState, password: ans.target.value}) }}
            />
            <button 
              type="button"
              title={isPasswordVisible ? "Скрыть пароль" : "Показать пароль"}
              className="cursor-pointer text-gray-400 rounded-e-md focus:outline-none focus-visible:text-blue-500 hover:text-blue-500 transition-colors" onClick={togglePasswordVisibility} aria-label={isPasswordVisible ? "Hide password" : "Show password"} aria-pressed={isPasswordVisible} aria-controls="password" >
              {isPasswordVisible ? ( <EyeOff size={20} aria-hidden="true" /> ) : ( <Eye size={20} aria-hidden="true" /> )}
            </button>
          </div>

          {userInfo.error &&
            <div 
              className="text-red-500 animate-slideIn opacity-0"
              style={{ "--delay": 0 + "s" }}
            >
              {userInfo.error}
            </div>
          }

          <button
            type="submit"
            className="w-full rounded-lg px-4 py-3 text-white font-semibold transition bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400"
            disabled={!(formState.login && formState.password) || userInfo.error}
          >
            Войти
          </button>

          <div>
            <input
              type="checkbox"
              id="rememberMe"
              className="relative top-[1px]"
              onClick={() => {
                dispatch(toggleRememberMe())
              }}
            />
            <label
              className="ml-[3px]"
            >
              Запомнить меня
            </label>
          </div>

          <div className="flex flex-row items-center justify-center">
            <span className="block truncate white">Еще не зарегистрированы?</span>
            <Link
              to={SIGN_UP}
              className="text-blue-600 cursor-pointer"
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