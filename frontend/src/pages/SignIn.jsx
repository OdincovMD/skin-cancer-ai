import React, { useEffect, useState } from "react"
import { useDispatch, useSelector } from "react-redux"
import { Link, useNavigate } from "react-router-dom"
import { Eye, EyeOff, LogIn, Scan } from "lucide-react"

import Alert from "../components/ui/Alert"
import Button from "../components/ui/Button"
import { onVerify } from "../asyncActions/onVerify"
import { HOME, SIGN_IN, SIGN_UP } from "../imports/ENDPOINTS"
import { mappingInfo } from "../imports/HELPERS"
import { noError, toggleRememberMe } from "../store/userReducer"

const SignIn = () => {
  const [formState, setFormState] = useState({ login: "", password: "" })
  const [isRequestPending, setIsRequestPending] = useState(false)
  const [isPasswordVisible, setIsPasswordVisible] = useState(false)

  const dispatch = useDispatch()
  const navigate = useNavigate()
  const userInfo = useSelector((state) => state.user)

  useEffect(() => {
    if (!isRequestPending && userInfo.error) dispatch(noError())
  }, [formState])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setIsRequestPending(true)
    try {
      const result = await dispatch(
        onVerify({
          data: {
            [mappingInfo.LOGIN]: formState.login,
            [mappingInfo.PASSWORD]: formState.password,
          },
          endpoint: SIGN_IN,
        })
      ).unwrap()
      if (!result.error && result.userData?.id != null) {
        navigate(HOME, { replace: true })
      }
    } finally {
      setIsRequestPending(false)
    }
  }

  const canSubmit = formState.login && formState.password && !isRequestPending

  return (
    <div className="flex min-h-[calc(100vh-3.5rem)] items-center justify-center px-4 py-12">
      <div className="w-full max-w-sm">
        <div className="mb-8 flex flex-col items-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-med-600 text-white mb-4">
            <Scan size={24} />
          </div>
          <h1 className="text-2xl font-bold text-gray-900">Вход в систему</h1>
          <p className="mt-1 text-sm text-gray-500">
            Введите данные для доступа к классификатору
          </p>
        </div>

        <div className="card-elevated">
          <form className="space-y-5" onSubmit={handleSubmit}>
            <div>
              <label htmlFor="login" className="input-label">
                Логин
              </label>
              <input
                id="login"
                type="text"
                autoComplete="username"
                placeholder="Ваш логин"
                className="input-field"
                value={formState.login}
                onChange={(e) =>
                  setFormState({ ...formState, login: e.target.value })
                }
              />
            </div>

            <div>
              <label htmlFor="password" className="input-label">
                Пароль
              </label>
              <div className="relative">
                <input
                  id="password"
                  type={isPasswordVisible ? "text" : "password"}
                  autoComplete="current-password"
                  placeholder="Введите пароль"
                  className="input-field pr-10"
                  value={formState.password}
                  onChange={(e) =>
                    setFormState({ ...formState, password: e.target.value })
                  }
                />
                <button
                  type="button"
                  onClick={() => setIsPasswordVisible((v) => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                  aria-label={
                    isPasswordVisible ? "Скрыть пароль" : "Показать пароль"
                  }
                >
                  {isPasswordVisible ? (
                    <EyeOff size={18} />
                  ) : (
                    <Eye size={18} />
                  )}
                </button>
              </div>
            </div>

            <label className="flex items-center gap-2.5 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={userInfo.isRememberMeChecked}
                onChange={() => dispatch(toggleRememberMe())}
                className="h-4 w-4 rounded border-gray-300 text-med-600 focus:ring-med-500"
              />
              <span className="text-sm text-gray-600">Запомнить меня</span>
            </label>

            {userInfo.error && (
              <Alert variant="error" className="animate-slideIn">
                {userInfo.error}
              </Alert>
            )}

            <Button type="submit" disabled={!canSubmit} className="w-full">
              {isRequestPending ? (
                <>
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  Вход...
                </>
              ) : (
                <>
                  <LogIn size={18} />
                  Войти
                </>
              )}
            </Button>
          </form>
        </div>

        <p className="mt-6 text-center text-sm text-gray-500">
          Ещё нет аккаунта?{" "}
          <Link to={SIGN_UP} className="text-link">
            Зарегистрироваться
          </Link>
        </p>
      </div>
    </div>
  )
}

export default SignIn
