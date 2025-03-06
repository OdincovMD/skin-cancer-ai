import React, { useState } from 'react'
import {useDispatch, useSelector} from "react-redux"
import { Link } from "react-router-dom"

import { onSignUp } from "../asyncActions/onSignUp"

const SignUp = () => {

  const dispatch = useDispatch()

  const [password, setPassword] = useState(null)
  const [repPassword, setRepPassword] = useState(null)
  var [name, surname, email, login, myForm] = [null, null, null, null, null]

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-100 p-6">
      <div className="w-full max-w-md rounded-2xl bg-white p-8 shadow-lg">
        <h2 className="mb-6 text-center text-2xl font-semibold text-gray-700">Регистрация</h2>
        <form className="space-y-4" ref={(el) => {myForm = el}}>
          <input 
            type="text" 
            placeholder="Имя" 
            className="w-full rounded-lg border border-gray-300 p-3 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            required
            onChange={(ans) => { name = ans.target.value }}
          />
          <input 
            type="text" 
            placeholder="Фамилия" 
            className="w-full rounded-lg border border-gray-300 p-3 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            required
            onChange={(ans) => { surname = ans.target.value }}
          />
          <input 
            type="text" 
            placeholder="Логин" 
            className="w-full rounded-lg border border-gray-300 p-3 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            required
            onChange={(ans) => { login = ans.target.value }}
          />
          <input 
            type="email" 
            placeholder="Электронная почта" 
            className="w-full rounded-lg border border-gray-300 p-3 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            required
            onChange={(ans) => { email = ans.target.value }}
          />
          <input 
            type="password" 
            placeholder="Пароль" 
            className="w-full rounded-lg border border-gray-300 p-3 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            pattern="[0-9A-z]{8,}"
            required
            onChange={(ans) => { setPassword(ans.target.value ) }}
          />
          <input 
            type="password" 
            placeholder="Подтвердите пароль" 
            className="w-full rounded-lg border border-gray-300 p-3 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            onChange={(ans) => { setRepPassword( ans.target.value ) }}
          />
          <p>{(password != repPassword) && 'Пароли не совпадают'}</p>

          {/* <div className="flex items-center gap-2">
            <input id="allowEmail" type="checkbox" className="h-5 w-5 rounded border-gray-300 text-blue-600 focus:ring-blue-500" />
            <label htmlFor="allowEmail" className="text-gray-600">I want to join the newsletter</label>
          </div> */}
          
          <button 
            type="submit" 
            className="w-full rounded-lg bg-blue-600 px-4 py-3 text-white font-semibold transition hover:bg-blue-700"
            disabled={!(password == repPassword)}
            onClick={ () => {
              dispatch(onSignUp({name: name, email: email, login: login, password: password}))
              myForm.reset()
              [name, email, login, myForm] = [null, null, null, null]
              setPassword(null)
              setRepPassword(null)
          }}
          >
            Зарегестрироваться
          </button>
          <div className="flex flex-row items-center justify-center">
            <span className="block truncate white">Уже зарегестрированы?</span>
            <Link
              to={'/signin'}
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