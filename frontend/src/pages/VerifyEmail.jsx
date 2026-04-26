import React, { useEffect, useState } from "react"
import { useSearchParams } from "react-router-dom"
import { useDispatch } from "react-redux"
import { CheckCircle2, Scan, XCircle } from "lucide-react"

import { fetchSessionMe } from "../asyncActions/fetchSessionMe"
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
    <div className="flex h-[calc(100vh-3.5rem)] overflow-hidden items-center justify-center px-4 py-4 sm:py-6">
      <div className="max-h-full w-full max-w-sm overflow-hidden rounded-2xl shadow-xl ring-1 ring-gray-900/[0.07]">

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
                Подтверждение email
              </h1>
            </div>
          </div>
        </div>

        {/* Phase content */}
        <div className="bg-white px-6 py-8 text-center">

          {phase === "loading" && (
            <div className="flex flex-col items-center gap-4">
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-med-50 ring-1 ring-med-100">
                <Spinner size="md" />
              </div>
              <div>
                <p className="font-semibold text-gray-800">Проверяем ссылку…</p>
                <p className="mt-1 text-sm text-gray-500">Это займёт несколько секунд</p>
              </div>
            </div>
          )}

          {phase === "ok" && (
            <div className="flex flex-col items-center gap-5">
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-green-50 ring-1 ring-green-100">
                <CheckCircle2 size={32} className="text-green-500" />
              </div>
              <div>
                <p className="text-lg font-bold text-gray-900">Email подтверждён!</p>
                <p className="mt-1.5 text-sm leading-relaxed text-gray-500">{detail}</p>
              </div>
              <div className="flex w-full flex-col gap-2.5 sm:flex-row sm:justify-center">
                <Button variant="primary" to={PROFILE} className="sm:min-w-[9rem]">
                  Личный кабинет
                </Button>
                <Button variant="secondary" to={SIGN_IN} className="sm:min-w-[9rem]">
                  Войти
                </Button>
              </div>
            </div>
          )}

          {phase === "error" && (
            <div className="flex flex-col items-center gap-5">
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-red-50 ring-1 ring-red-100">
                <XCircle size={32} className="text-red-500" />
              </div>
              <div>
                <p className="text-lg font-bold text-gray-900">Не удалось подтвердить</p>
                <p className="mt-1.5 text-sm leading-relaxed text-gray-500">{detail}</p>
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

export default VerifyEmail
