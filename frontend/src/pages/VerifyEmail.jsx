import React, { useEffect, useState } from "react"
import { useSearchParams } from "react-router-dom"
import { useDispatch } from "react-redux"
import { Mail, Scan } from "lucide-react"

import { fetchSessionMe } from "../asyncActions/fetchSessionMe"
import Alert from "../components/ui/Alert"
import Button from "../components/ui/Button"
import Spinner from "../components/ui/Spinner"
import { env } from "../imports/ENV"
import { PROFILE, SIGN_IN, VERIFY_EMAIL } from "../imports/ENDPOINTS"
import { notifyOtherTabsSessionMayHaveChanged } from "../imports/sessionSync"

const VerifyEmail = () => {
  const dispatch = useDispatch()
  const [searchParams] = useSearchParams()
  const token = searchParams.get("token")
  const [phase, setPhase] = useState("loading")
  const [detail, setDetail] = useState("")

  useEffect(() => {
    if (!token) {
      setPhase("error")
      setDetail(
        "В ссылке отсутствует токен. Откройте письмо ещё раз или запросите новую регистрацию."
      )
      return
    }

    const okKey = `skin_email_verified_${token}`
    if (sessionStorage.getItem(okKey) === "1") {
      setPhase("ok")
      setDetail(
        "Адрес подтверждён. Откройте основную вкладку с сайтом — статус подтверждения обновится сам."
      )
      notifyOtherTabsSessionMayHaveChanged()
      dispatch(fetchSessionMe())
      return
    }

    const base = env.BACKEND_URL.replace(/\/$/, "")
    const url = `${base}${VERIFY_EMAIL}?token=${encodeURIComponent(token)}`
    const ac = new AbortController()

    fetch(url, {
      method: "GET",
      headers: { accept: "application/json" },
      signal: ac.signal,
    })
      .then(async (res) => {
        const data = await res.json().catch(() => ({}))
        if (res.ok && data.ok === true) {
          sessionStorage.setItem(okKey, "1")
          setPhase("ok")
          setDetail(
            "Адрес подтверждён. Откройте основную вкладку с сайтом — статус подтверждения обновится сам."
          )
          notifyOtherTabsSessionMayHaveChanged()
          dispatch(fetchSessionMe())
          return
        }
        if (!res.ok) {
          if (sessionStorage.getItem(okKey) === "1") return
          setPhase("error")
          setDetail("Не удалось связаться с сервером. Попробуйте позже.")
          return
        }
        if (sessionStorage.getItem(okKey) === "1") return
        setPhase("error")
        setDetail(
          typeof data.error === "string" && data.error
            ? data.error
            : "Не удалось подтвердить email."
        )
      })
      .catch((err) => {
        if (err?.name === "AbortError") return
        if (sessionStorage.getItem(okKey) === "1") return
        setPhase("error")
        setDetail("Ошибка сети. Попробуйте позже.")
      })

    return () => ac.abort()
  }, [token, dispatch])

  return (
    <div className="flex min-h-[calc(100vh-3.5rem)] items-center justify-center px-4 py-12">
      <div className="w-full max-w-sm">
        <div className="mb-8 flex flex-col items-center text-center">
          <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-med-600 text-white">
            <Scan size={24} />
          </div>
          <div className="flex items-center justify-center gap-2 text-gray-900">
            <Mail size={20} className="text-med-600" />
            <h1 className="text-2xl font-bold">Подтверждение email</h1>
          </div>
        </div>

        <div className="card-elevated text-center">
          {phase === "loading" && (
            <div className="flex flex-col items-center gap-4 py-4">
              <Spinner size="md" />
              <p className="text-sm text-gray-600">Проверяем ссылку…</p>
            </div>
          )}

          {phase === "ok" && (
            <div className="space-y-6">
              <Alert variant="success" title="Готово">
                {detail}
              </Alert>
              <div className="flex flex-col gap-3 sm:flex-row sm:justify-center">
                <Button variant="primary" to={PROFILE} className="sm:min-w-[10rem]">
                  Личный кабинет
                </Button>
                <Button variant="secondary" to={SIGN_IN} className="sm:min-w-[10rem]">
                  Вход
                </Button>
              </div>
            </div>
          )}

          {phase === "error" && (
            <div className="space-y-6">
              <Alert variant="error" title="Не удалось подтвердить">
                {detail}
              </Alert>
              <Button variant="secondary" to={SIGN_IN} className="w-full">
                На страницу входа
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default VerifyEmail
