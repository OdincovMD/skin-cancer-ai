import React, { useState } from "react"
import { Link } from "react-router-dom"
import { CheckCircle2, KeyRound, Scan } from "lucide-react"

import Alert from "../components/ui/Alert"
import Button from "../components/ui/Button"
import { env } from "../imports/ENV"
import { FORGOT_PASSWORD_API, SIGN_IN } from "../imports/ENDPOINTS"

const ForgotPassword = () => {
  const [email, setEmail] = useState("")
  const [isPending, setIsPending] = useState(false)
  const [phase, setPhase] = useState("form") // "form" | "sent"
  const [error, setError] = useState("")

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError("")
    setIsPending(true)
    try {
      const res = await fetch(`${env.BACKEND_URL}${FORGOT_PASSWORD_API}`, {
        method: "POST",
        headers: { "Content-Type": "application/json", accept: "application/json" },
        body: JSON.stringify({ email: email.trim() }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        const d = data?.detail
        setError(
          typeof d === "string"
            ? d
            : typeof data?.error === "string"
            ? data.error
            : "Ошибка сервера. Попробуйте позже."
        )
        return
      }
      if (data?.error) {
        setError(data.error)
        return
      }
      setPhase("sent")
    } catch {
      setError("Ошибка сети. Проверьте соединение и попробуйте снова.")
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
                Восстановление пароля
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
                  Введите адрес электронной почты, привязанный к вашему аккаунту. Мы отправим ссылку для сброса пароля.
                </p>
              </div>

              <form className="space-y-4" onSubmit={handleSubmit}>
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
                    value={email}
                    onChange={(e) => {
                      setEmail(e.target.value)
                      if (error) setError("")
                    }}
                  />
                </div>

                {error && (
                  <Alert variant="error" className="animate-slideIn">
                    {error}
                  </Alert>
                )}

                <Button type="submit" disabled={!email.trim() || isPending} className="w-full">
                  {isPending ? (
                    <>
                      <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                      Отправка...
                    </>
                  ) : (
                    <>
                      <KeyRound size={17} />
                      Отправить ссылку
                    </>
                  )}
                </Button>
              </form>

              <p className="mt-6 text-center text-sm text-gray-500">
                Вспомнили пароль?{" "}
                <Link to={SIGN_IN} className="text-link">
                  Войти
                </Link>
              </p>
            </>
          )}

          {phase === "sent" && (
            <div className="flex flex-col items-center gap-5 text-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-green-50 ring-1 ring-green-100">
                <CheckCircle2 size={32} className="text-green-500" />
              </div>
              <div>
                <p className="text-lg font-bold text-gray-900">Письмо отправлено</p>
                <p className="mt-1.5 text-sm leading-relaxed text-gray-500">
                  Если аккаунт с адресом <strong className="text-gray-700">{email}</strong> существует, письмо со ссылкой для сброса пароля уже в пути. Проверьте папку «Спам», если письмо не пришло.
                </p>
              </div>
              <Button variant="secondary" to={SIGN_IN} className="w-full sm:w-auto sm:min-w-[12rem]">
                На страницу входа
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ForgotPassword
