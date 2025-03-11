import React, { useEffect, useState } from "react"
import { useDispatch, useSelector } from "react-redux"
import { Link, useNavigate } from "react-router-dom"
import { Eye, EyeOff } from "lucide-react"

import { onVerify } from "../asyncActions/onVerify"
import { HOME, SIGN_IN, SIGN_UP } from "../imports/ENDPOINTS"
import { mappingInfo } from "../imports/HELPERS"

const SignUp = () => {

  const defaultFormState = {
    firstName: null,
    lastName: null,
    login: null,
    email: null,
    password: null,
    repPassword: null
  }

  const [formState, setFormState] = useState(defaultFormState)

  const dispatch = useDispatch()
  const userInfo = useSelector(state => state.user)
  const navigate = useNavigate()

  const [isPasswordVisible, setIsPasswordVisible] = useState(false)
  const [isPasswordValid, setIsPasswordValid] = useState(false)
  const [arePasswordsSame, setArePasswordsSame] = useState(false)

  const togglePasswordVisibility = () => {
    setIsPasswordVisible((prevState) => !prevState)
  }
  
  const handleSubmit = async (event) => {
    event.preventDefault()
    event.target.reset()

    dispatch(onVerify({ 
      data: { 
        [mappingInfo.FIRST_NAME]: formState.firstName,
        [mappingInfo.LAST_NAME]: formState.lastName,
        [mappingInfo.EMAIL]: formState.email,
        [mappingInfo.LOGIN]: formState.login,
        [mappingInfo.PASSWORD]: formState.password
      },
       endpoint: SIGN_UP 
    }))
    setFormState(defaultFormState)
    userInfo.userData.id && navigate(HOME)
  }

  useEffect(() => {

    formState.password && setIsPasswordValid(formState.password.match(/[0-9A-z]{8,}/)),
    setArePasswordsSame(formState.password == formState.repPassword)

  }, [formState])

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-100 p-6">
      <div className="w-full max-w-md rounded-2xl bg-white p-8 shadow-lg">
        <h2 className="mb-6 text-center text-2xl font-semibold text-gray-700">Регистрация</h2>
        <form className="space-y-4" onSubmit={handleSubmit}>

          <input 
            type="text" 
            placeholder="Имя" 
            className="w-full rounded-lg border border-gray-300 p-3 outline-blue-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            required
            onChange={(ans) => { setFormState({...formState, firstName: ans.target.value}) }}
          />

          <input 
            type="text" 
            placeholder="Фамилия" 
            className="w-full rounded-lg border border-gray-300 p-3 outline-blue-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            required
            onChange={(ans) => { setFormState({...formState, lastName: ans.target.value}) }}
          />

          <input 
            type="text" 
            placeholder="Логин" 
            className="w-full rounded-lg border border-gray-300 p-3 outline-blue-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            required
            onChange={(ans) => { setFormState({...formState, login: ans.target.value}) }}
          />

          <input 
            type="email" 
            placeholder="Электронная почта"
            className="w-full rounded-lg border border-gray-300 p-3 outline-blue-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            required
            onChange={(ans) => { setFormState({...formState, email: ans.target.value}) }}
          />
          
          <div
            className="flex flex-row w-full rounded-lg border border-gray-300 p-3 focus-within:border-blue-500 focus-within:ring-1 focus-within:ring-blue-500"
          >
            <input 
              type={isPasswordVisible ? "text" : "password"}
              placeholder="Пароль" 
              id="password-input"
              className="w-full border-none focus:outline-none"
              pattern="[0-9A-z]{8,}"
              required
              onChange={(ans) => { setFormState({...formState, password: ans.target.value}) }}
            />
            <button 
              type="button"
              title={isPasswordVisible ? "Скрыть пароль" : "Показать пароль"}
              className="cursor-pointer text-gray-400 rounded-e-md focus:outline-none focus-visible:text-blue-500 hover:text-blue-500 transition-colors" onClick={togglePasswordVisibility} aria-label={isPasswordVisible ? "Hide password" : "Show password"} aria-pressed={isPasswordVisible} aria-controls="password" >
              {isPasswordVisible ? ( <EyeOff size={20} aria-hidden="true" /> ) : ( <Eye size={20} aria-hidden="true" /> )}
            </button>
          </div>
          { formState.password && !isPasswordValid &&
              <p 
                className="text-gray-400 text-sm animate-slideIn opacity-0"
                style={{ "--delay": 0 + "s" }}
              >
                Пароль должен содержать только цифры и буквы латиницы, минимальная длина - 8 символов.
              </p>
          }

          <input 
            type="password" 
            placeholder="Подтвердите пароль" 
            className="w-full rounded-lg border border-gray-300 p-3 outline-blue-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            onChange={(ans) => { setFormState({...formState, repPassword: ans.target.value}) }}
          />

          {!arePasswordsSame &&
            <p
              className="text-gray-400 animate-slideIn opacity-0"
              style={{ "--delay": 0 + "s" }}
            >
              Пароли не совпадают
            </p>
          }

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
            disabled={!(formState.firstName && formState.lastName && formState.login && formState.email && formState.password && isPasswordValid && arePasswordsSame)}
          >
            Зарегестрироваться
          </button>

          <div className="flex flex-row items-center justify-center">
            <span className="block truncate white">Уже зарегестрированы?</span>
            <Link
              to={SIGN_IN}
              className="text-blue-600"
            >
              <span className="underline ml-1 transition hover:text-blue-700">{`Вход`}</span>
           </Link>
          </div>
        </form>
      </div>
    </div>
  )
}

export default SignUp