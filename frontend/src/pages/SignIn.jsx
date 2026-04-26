import React, { useEffect, useState } from "react"
import { useDispatch, useSelector } from "react-redux"
import { Link, useNavigate } from "react-router-dom"
import { CheckCircle2, Eye, EyeOff, LogIn, Scan } from "lucide-react"

import Alert from "../components/ui/Alert"
import Button from "../components/ui/Button"
import { onVerify } from "../asyncActions/onVerify"
import { HOME, SIGN_IN, SIGN_UP } from "../imports/ENDPOINTS"
import { mappingInfo } from "../imports/HELPERS"
import { noError, toggleRememberMe } from "../store/userReducer"

const BRAND_FEATURES = [
  "Метод Киттлера для структурного анализа",
  "Раннее выявление подозрительных признаков",
  "Прозрачная логика принятия решений",
]

const SignIn = () => {
  const [formState, setFormState] = useState({ login: "", password: "" })
  const [isRequestPending, setIsRequestPending] = useState(false)
  const [isPasswordVisible, setIsPasswordVisible] = useState(false)

  const dispatch = useDispatch()
  const navigate = useNavigate()
  const userInfo = useSelector((state) => state.user)

  useEffect(() => {
    const previousBodyOverflow = document.body.style.overflow
    const previousHtmlOverflow = document.documentElement.style.overflow
    document.body.style.overflow = "hidden"
    document.documentElement.style.overflow = "hidden"

    return () => {
      document.body.style.overflow = previousBodyOverflow
      document.documentElement.style.overflow = previousHtmlOverflow
    }
  }, [])

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
    <div className="flex h-[calc(100vh-3.5rem)] overflow-hidden items-center justify-center px-4 py-4 sm:py-6">
      <div className="max-h-full w-full max-w-3xl overflow-hidden rounded-2xl shadow-xl ring-1 ring-gray-900/[0.07]">
        <div className="grid lg:grid-cols-[288px_1fr]">

          {/* ── Brand panel (desktop only) ── */}
          <div className="relative hidden flex-col justify-between overflow-hidden bg-gradient-to-br from-med-900 to-med-600 px-8 py-10 text-white lg:flex">
            <div className="absolute -right-12 -top-12 h-44 w-44 rounded-full bg-white/[0.06]" />
            <div className="absolute -bottom-16 -left-8 h-52 w-52 rounded-full bg-white/[0.06]" />
            <div className="absolute bottom-8 right-6 opacity-[0.06]">
              <Scan size={108} />
            </div>

            <div className="relative">
              <div className="mb-5 flex h-11 w-11 items-center justify-center rounded-xl bg-white/15 ring-1 ring-white/20">
                <Scan size={22} />
              </div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-med-300">
                Skin Cancer AI
              </p>
              <p className="mt-2 text-sm leading-relaxed text-white/65">
                Дерматоскопическая диагностика на базе машинного обучения и метода Киттлера.
              </p>
            </div>
          </div>

          {/* ── Form panel ── */}
          <div className="bg-white px-8 py-10">
            {/* Mobile branding */}
            <div className="mb-6 flex items-center gap-3 lg:hidden">
              <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl bg-med-600 text-white">
                <Scan size={18} />
              </div>
              <span className="text-sm font-semibold text-gray-700">Skin Cancer AI</span>
            </div>

            <h1 className="text-2xl font-bold text-gray-900">Вход в систему</h1>
            <p className="mt-1 text-sm text-gray-500">
              Введите данные для доступа к классификатору
            </p>

            <form className="mt-8 space-y-5" onSubmit={handleSubmit}>
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
                    aria-label={isPasswordVisible ? "Скрыть пароль" : "Показать пароль"}
                  >
                    {isPasswordVisible ? <EyeOff size={18} /> : <Eye size={18} />}
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

            <p className="mt-6 text-center text-sm text-gray-500">
              Ещё нет аккаунта?{" "}
              <Link to={SIGN_UP} className="text-link">
                Зарегистрироваться
              </Link>
            </p>
          </div>

        </div>
      </div>
    </div>
  )
}

export default SignIn
