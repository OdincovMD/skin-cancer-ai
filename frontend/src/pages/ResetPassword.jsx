import React, { useEffect, useState } from "react"
import { useSearchParams } from "react-router-dom"
import { CheckCircle2, Eye, EyeOff, KeyRound, Scan, XCircle } from "lucide-react"

import Alert from "../components/ui/Alert"
import Button from "../components/ui/Button"
import Spinner from "../components/ui/Spinner"
import { env } from "../imports/ENV"
import { FORGOT_PASSWORD, RESET_PASSWORD_API, SIGN_IN } from "../imports/ENDPOINTS"

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

const ResetPassword = () => {
  const [searchParams] = useSearchParams()
  const token = searchParams.get("token")

  const [phase, setPhase] = useState("form") // "form" | "success" | "error"
  const [errorMsg, setErrorMsg] = useState("")
  const [formError, setFormError] = useState("")
  const [isPending, setIsPending] = useState(false)
  const [newPassword, setNewPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [isPasswordVisible, setIsPasswordVisible] = useState(false)

  useEffect(() => {
    if (!token) {
      setPhase("error")
      setErrorMsg("В ссылке отсутствует токен. Проверьте письмо или запросите новую ссылку.")
    }
  }, [token])

  const pwdLenOk = newPassword.length >= 8
  const arePasswordsSame = newPassword === confirmPassword && confirmPassword.length > 0

  const canSubmit =
    !isPending &&
    token &&
    newPassword &&
    confirmPassword &&
    pwdLenOk &&
    arePasswordsSame

  const handleSubmit = async (e) => {
    e.preventDefault()
    setFormError("")
    setIsPending(true)
    try {
      const res = await fetch(`${env.BACKEND_URL}${RESET_PASSWORD_API}`, {
        method: "POST",
        headers: { "Content-Type": "application/json", accept: "application/json" },
        body: JSON.stringify({ token, new_password: newPassword }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        const d = data?.detail
        setFormError(
          typeof d === "string"
            ? d
            : typeof data?.error === "string"
            ? data.error
            : "Ошибка сервера. Попробуйте позже."
        )
        return
      }
      if (data?.ok === true) {
        setPhase("success")
        return
      }
      setFormError(
        typeof data?.error === "string" && data.error
          ? data.error
          : "Не удалось сбросить пароль."
      )
    } catch {
      setFormError("Ошибка сети. Проверьте соединение и попробуйте снова.")
    } finally {
      setIsPending(false)
    }
  }

  return (
    <div className="flex min-h-[calc(100vh-3.5rem)] items-center justify-center px-4 py-6">
      <div className="w-full max-w-sm rounded-2xl shadow-xl ring-1 ring-gray-900/[0.07]">

        {/* Colored header band */}
        <div className="bg-gradient-to-r from-med-900 to-med-600 px-6 py-5">
          <div className="flex items-center gap-3 text-white">
            <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl bg-white/15 ring-1 ring-white/20">
              <Scan size={18} />
            </div>
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-med-300">
                Skin Cancer AI
              </p>
              <h1 className="text-lg font-bold leading-tight">
                Новый пароль
              </h1>
            </div>
          </div>
        </div>

        <div className="bg-white px-6 py-8">

          {phase === "form" && (
            <>
              <div className="flex flex-col items-center gap-3 mb-6 text-center">
                <div className="flex h-14 w-14 items-center justify-center rounded-full bg-med-50 ring-1 ring-med-100">
                  <KeyRound size={26} className="text-med-600" />
                </div>
                <p className="text-sm leading-relaxed text-gray-500 max-w-xs">
                  Придумайте новый пароль для вашего аккаунта.
                </p>
              </div>

              <form className="space-y-4" onSubmit={handleSubmit}>
                <div>
                  <label htmlFor="new-password" className="input-label">
                    Новый пароль
                  </label>
                  <div className="relative">
                    <input
                      id="new-password"
                      type={isPasswordVisible ? "text" : "password"}
                      autoComplete="new-password"
                      placeholder="Минимум 8 символов"
                      required
                      className="input-field pr-10"
                      value={newPassword}
                      onChange={(e) => {
                        setNewPassword(e.target.value)
                        if (formError) setFormError("")
                      }}
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
                  {newPassword && (
                    <ul className="mt-2 space-y-1">
                      <PasswordHint ok={pwdLenOk} text="Не менее 8 символов" />
                    </ul>
                  )}
                </div>

                <div>
                  <label htmlFor="confirm-password" className="input-label">
                    Подтверждение пароля
                  </label>
                  <input
                    id="confirm-password"
                    type="password"
                    autoComplete="new-password"
                    placeholder="Повторите пароль"
                    required
                    className="input-field"
                    value={confirmPassword}
                    onChange={(e) => {
                      setConfirmPassword(e.target.value)
                      if (formError) setFormError("")
                    }}
                  />
                  {confirmPassword && !arePasswordsSame && (
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

                {formError && (
                  <Alert variant="error" className="animate-slideIn">
                    {formError}
                  </Alert>
                )}

                <Button type="submit" disabled={!canSubmit} className="w-full">
                  {isPending ? (
                    <>
                      <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                      Сохранение...
                    </>
                  ) : (
                    <>
                      <KeyRound size={17} />
                      Установить пароль
                    </>
                  )}
                </Button>
              </form>
            </>
          )}

          {phase === "success" && (
            <div className="flex flex-col items-center gap-5 text-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-green-50 ring-1 ring-green-100">
                <CheckCircle2 size={32} className="text-green-500" />
              </div>
              <div>
                <p className="text-lg font-bold text-gray-900">Пароль изменён!</p>
                <p className="mt-1.5 text-sm leading-relaxed text-gray-500">
                  Новый пароль успешно установлен. Теперь вы можете войти в систему.
                </p>
              </div>
              <Button variant="primary" to={SIGN_IN} className="w-full sm:w-auto sm:min-w-[12rem]">
                Войти
              </Button>
            </div>
          )}

          {phase === "error" && (
            <div className="flex flex-col items-center gap-5 text-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-red-50 ring-1 ring-red-100">
                <XCircle size={32} className="text-red-500" />
              </div>
              <div>
                <p className="text-lg font-bold text-gray-900">Недействительная ссылка</p>
                <p className="mt-1.5 text-sm leading-relaxed text-gray-500">{errorMsg}</p>
              </div>
              <Button variant="secondary" to={FORGOT_PASSWORD} className="w-full sm:w-auto sm:min-w-[12rem]">
                Запросить новую ссылку
              </Button>
            </div>
          )}

        </div>
      </div>
    </div>
  )
}

export default ResetPassword
