import React, { useEffect, useState } from "react"
import { useDispatch, useSelector } from "react-redux"
import { Link, useNavigate } from "react-router-dom"
import {
  Eye,
  EyeOff,
  UserPlus,
  CheckCircle2,
  XCircle,
  Scan,
} from "lucide-react"

import Alert from "../components/ui/Alert"
import Button from "../components/ui/Button"
import { onVerify } from "../asyncActions/onVerify"
import { SIGN_IN, SIGN_UP, PROFILE } from "../imports/ENDPOINTS"
import { mappingInfo } from "../imports/HELPERS"
import { noError } from "../store/userReducer"

const BRAND_FEATURES = [
  "Безопасное хранение данных",
  "История всех проведённых анализов",
  "Быстрый старт без лишних шагов",
]

const PasswordHint = ({ ok, text }) => (
  <li className="flex items-center gap-1.5 text-xs">
    {ok ? (
      <CheckCircle2 size={14} className="text-green-500 flex-shrink-0" />
    ) : (
      <XCircle size={14} className="text-gray-300 flex-shrink-0" />
    )}
    <span className={ok ? "text-green-700" : "text-gray-500"}>{text}</span>
  </li>
)

const SignUp = () => {
  const defaultFormState = {
    [mappingInfo.FIRST_NAME]: "",
    [mappingInfo.LAST_NAME]: "",
    [mappingInfo.LOGIN]: "",
    [mappingInfo.EMAIL]: "",
    [mappingInfo.PASSWORD]: "",
    [mappingInfo.REP_PASSWORD]: "",
  }

  const [isRequestPending, setIsRequestPending] = useState(false)
  const [formState, setFormState] = useState(defaultFormState)
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

  const pwdLenOk = formState.password.length >= 8
  const pwdCharsOk = /^[0-9A-Za-z]+$/.test(formState.password)
  const isPasswordValid = pwdLenOk && pwdCharsOk
  const arePasswordsSame =
    formState.password === formState.repPassword &&
    formState.repPassword.length > 0

  const handleSubmit = async (e) => {
    e.preventDefault()
    setIsRequestPending(true)
    try {
      const outcome = await dispatch(
        onVerify({
          data: {
            [mappingInfo.FIRST_NAME]: formState.firstName,
            [mappingInfo.LAST_NAME]: formState.lastName,
            [mappingInfo.EMAIL]: formState.email,
            [mappingInfo.LOGIN]: formState.login,
            [mappingInfo.PASSWORD]: formState.password,
          },
          endpoint: SIGN_UP,
        })
      ).unwrap()

      if (!outcome.error && outcome.accessToken) {
        navigate(PROFILE, { replace: true })
      }
    } finally {
      setIsRequestPending(false)
    }
  }

  const allFieldsFilled =
    formState.firstName &&
    formState.lastName &&
    formState.login &&
    formState.email &&
    formState.password

  const canSubmit =
    !isRequestPending &&
    allFieldsFilled &&
    isPasswordValid &&
    arePasswordsSame

  const set = (field) => (e) =>
    setFormState({ ...formState, [field]: e.target.value })

  return (
    <div className="flex h-[calc(100vh-3.5rem)] overflow-hidden items-center justify-center px-4 py-4 sm:py-5">
      <div className="max-h-full w-full max-w-4xl overflow-hidden rounded-2xl shadow-xl ring-1 ring-gray-900/[0.07]">
        <div className="grid lg:grid-cols-[260px_1fr]">

          {/* ── Brand panel (desktop only) ── */}
          <div className="relative hidden flex-col justify-between overflow-hidden bg-gradient-to-br from-med-900 to-med-600 px-8 py-8 text-white lg:flex">
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
                Получите доступ к профессиональному дерматоскопическому анализатору.
              </p>
            </div>
          </div>

          {/* ── Form panel ── */}
          <div className="bg-white px-6 py-6 sm:px-8">
            {/* Mobile branding */}
            <div className="mb-6 flex items-center gap-3 lg:hidden">
              <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl bg-med-600 text-white">
                <Scan size={18} />
              </div>
              <span className="text-sm font-semibold text-gray-700">Skin Cancer AI</span>
            </div>

            <h1 className="text-2xl font-bold text-gray-900">Создание аккаунта</h1>
            <p className="mt-1 text-sm text-gray-500">
              Заполните данные для регистрации в системе
            </p>

            <form className="mt-5 space-y-4" onSubmit={handleSubmit}>

              {/* Personal data */}
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-gray-400">
                    Личные данные
                  </span>
                  <div className="h-px flex-1 bg-gray-100" />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label htmlFor="firstName" className="input-label">
                      Имя
                    </label>
                    <input
                      id="firstName"
                      type="text"
                      autoComplete="given-name"
                      placeholder="Иван"
                      required
                      className="input-field"
                      value={formState.firstName}
                      onChange={set("firstName")}
                    />
                  </div>
                  <div>
                    <label htmlFor="lastName" className="input-label">
                      Фамилия
                    </label>
                    <input
                      id="lastName"
                      type="text"
                      autoComplete="family-name"
                      placeholder="Иванов"
                      required
                      className="input-field"
                      value={formState.lastName}
                      onChange={set("lastName")}
                    />
                  </div>
                </div>

                <div>
                  <label htmlFor="email" className="input-label">
                    Электронная почта
                  </label>
                  <input
                    id="email"
                    type="email"
                    autoComplete="email"
                    placeholder="ivanov@example.com"
                    required
                    className="input-field"
                    value={formState.email}
                    onChange={set("email")}
                  />
                </div>
              </div>

              {/* Account data */}
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-gray-400">
                    Данные аккаунта
                  </span>
                  <div className="h-px flex-1 bg-gray-100" />
                </div>

                <div>
                  <label htmlFor="login" className="input-label">
                    Логин
                  </label>
                  <input
                    id="login"
                    type="text"
                    autoComplete="username"
                    placeholder="ivanov_doctor"
                    required
                    className="input-field"
                    value={formState.login}
                    onChange={set("login")}
                  />
                </div>

                <div>
                  <label htmlFor="reg-password" className="input-label">
                    Пароль
                  </label>
                  <div className="relative">
                    <input
                      id="reg-password"
                      type={isPasswordVisible ? "text" : "password"}
                      autoComplete="new-password"
                      placeholder="Минимум 8 символов"
                      required
                      className="input-field pr-10"
                      value={formState.password}
                      onChange={set("password")}
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

                  {formState.password && (
                    <ul className="mt-2 space-y-1">
                      <PasswordHint ok={pwdLenOk} text="Не менее 8 символов" />
                      <PasswordHint ok={pwdCharsOk} text="Только латинские буквы и цифры" />
                    </ul>
                  )}
                </div>

                <div>
                  <label htmlFor="reg-password-confirm" className="input-label">
                    Подтверждение пароля
                  </label>
                  <input
                    id="reg-password-confirm"
                    type="password"
                    autoComplete="new-password"
                    placeholder="Повторите пароль"
                    className="input-field"
                    value={formState.repPassword}
                    onChange={set("repPassword")}
                  />
                  {formState.repPassword && !arePasswordsSame && (
                    <p className="mt-1.5 flex items-center gap-1.5 text-xs text-red-600">
                      <XCircle size={14} className="flex-shrink-0" />
                      Пароли не совпадают
                    </p>
                  )}
                  {arePasswordsSame && (
                    <p className="mt-1.5 flex items-center gap-1.5 text-xs text-green-600">
                      <CheckCircle2 size={14} className="flex-shrink-0" />
                      Пароли совпадают
                    </p>
                  )}
                </div>
              </div>

              {userInfo.error && (
                <Alert variant="error" className="animate-slideIn">
                  {userInfo.error}
                </Alert>
              )}

              <Button type="submit" disabled={!canSubmit} className="w-full">
                {isRequestPending ? (
                  <>
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                    Регистрация...
                  </>
                ) : (
                  <>
                    <UserPlus size={18} />
                    Зарегистрироваться
                  </>
                )}
              </Button>
            </form>

            <p className="mt-4 text-center text-sm text-gray-500">
              Уже есть аккаунт?{" "}
              <Link to={SIGN_IN} className="text-link">
                Войти
              </Link>
            </p>
          </div>

        </div>
      </div>
    </div>
  )
}

export default SignUp
