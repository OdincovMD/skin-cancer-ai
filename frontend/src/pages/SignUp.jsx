import React, { useState } from 'react'
import { useDispatch } from "react-redux"
import { Link } from "react-router-dom"
import { Eye, EyeOff} from 'lucide-react'

import { onVerify } from "../asyncActions/onVerify"
import { SIGN_IN, SIGN_UP } from "../imports/ENDPOINTS"

const SignUp = () => {

  const dispatch = useDispatch()

  const [password, setPassword] = useState(null)
  const [repPassword, setRepPassword] = useState(null)
  const [isPasswordVisible, setIsPasswordVisible] = useState(false)
  var [firstName, lastName, email, login] = [null, null, null, null]

  const togglePasswordVisibility = () => {
    setIsPasswordVisible((prevState) => !prevState);
}

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-100 p-6">
      <div className="w-full max-w-md rounded-2xl bg-white p-8 shadow-lg">
        <h2 className="mb-6 text-center text-2xl font-semibold text-gray-700">Регистрация</h2>
        <form className="space-y-4">

          <input 
            type="text" 
            placeholder="Имя" 
            className="w-full rounded-lg border border-gray-300 p-3 outline-blue-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            required
            onChange={(ans) => { firstName = ans.target.value }}
          />

          <input 
            type="text" 
            placeholder="Фамилия" 
            className="w-full rounded-lg border border-gray-300 p-3 outline-blue-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            required
            onChange={(ans) => { lastName = ans.target.value }}
          />

          <input 
            type="text" 
            placeholder="Логин" 
            className="w-full rounded-lg border border-gray-300 p-3 outline-blue-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            required
            onChange={(ans) => { login = ans.target.value }}
          />

          <input 
            type="email" 
            placeholder="Электронная почта"
            pattern="[0-9A-z_\.]+@[A-z]{2,}\.[A-z]+"
            className="w-full rounded-lg border border-gray-300 p-3 outline-blue-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            required
            onChange={(ans) => { email = ans.target.value }}
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
              onChange={(ans) => { setPassword( ans.target.value ) }}
            />
            <button 
              type="button"
              title={isPasswordVisible ? "Скрыть пароль" : "Показать пароль"}
              className="cursor-pointer text-gray-400 rounded-e-md focus:outline-none focus-visible:text-blue-500 hover:text-blue-500 transition-colors" onClick={togglePasswordVisibility} aria-label={isPasswordVisible ? "Hide password" : "Show password"} aria-pressed={isPasswordVisible} aria-controls="password" >
              {isPasswordVisible ? ( <EyeOff size={20} aria-hidden="true" /> ) : ( <Eye size={20} aria-hidden="true" /> )}
            </button>
          </div>

          <input 
            type="password" 
            placeholder="Подтвердите пароль" 
            className="w-full rounded-lg border border-gray-300 p-3 outline-blue-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            onChange={(ans) => { setRepPassword( ans.target.value ) }}
          />

          <p>{(password != repPassword) && 'Пароли не совпадают'}</p>

          {/* <div className="flex items-center gap-2">
            <input id="allowEmail" type="checkbox" className="h-5 w-5 rounded border-gray-300 text-blue-600 focus:ring-blue-500" />
            <label htmlFor="allowEmail" className="text-gray-600">I want to join the newsletter</label>
          </div> */}
          
          <button 
            type="submit" 
            className={`w-full rounded-lg px-4 py-3 text-white font-semibold transition ${"bg-blue-600  hover:bg-blue-700"}`}
            disabled={!(password == repPassword)}
            onClick={ () => {
              dispatch(onVerify({firstName: firstName, lastName: lastName, email: email, login: login, password: password}, SIGN_UP))
              [firstName, email, login, myForm] = [null, null, null, null]
              setPassword(null)
              setRepPassword(null)
          }}
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