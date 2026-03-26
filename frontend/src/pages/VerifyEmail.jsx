import React, { useEffect, useState } from "react"
import { Link, useSearchParams } from "react-router-dom"
import { useDispatch } from "react-redux"

import { fetchSessionMe } from "../asyncActions/fetchSessionMe"
import { env } from "../imports/ENV"
import { PROFILE, SIGN_IN } from "../imports/ENDPOINTS"

const VerifyEmail = () => {
  const dispatch = useDispatch()
  const [searchParams] = useSearchParams()
  const token = searchParams.get("token")
  const [phase, setPhase] = useState("loading")
  const [detail, setDetail] = useState("")

  useEffect(() => {
    if (!token) {
      setPhase("error")
      setDetail("В ссылке отсутствует токен. Откройте письмо ещё раз или запросите новую регистрацию.")
      return
    }

    const base = env.BACKEND_URL.replace(/\/$/, "")
    const url = `${base}/verify-email?token=${encodeURIComponent(token)}`

    fetch(url, { method: "GET", headers: { accept: "application/json" } })
      .then(async (res) => {
        const data = await res.json().catch(() => ({}))
        if (data.ok) {
          setPhase("ok")
          setDetail("Адрес подтверждён. Если вы уже вошли в аккаунт в этом браузере, статус обновится автоматически.")
          dispatch(fetchSessionMe())
        } else {
          setPhase("error")
          setDetail(data.error || "Не удалось подтвердить email.")
        }
      })
      .catch(() => {
        setPhase("error")
        setDetail("Ошибка сети. Попробуйте позже.")
      })
  }, [token])

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-100 p-6">
      <div className="w-full max-w-md rounded-2xl bg-white p-8 shadow-lg text-center">
        <h1 className="mb-4 text-xl font-semibold text-gray-800">Подтверждение email</h1>
        {phase === "loading" && (
          <p className="text-gray-600">Проверяем ссылку…</p>
        )}
        {phase === "ok" && (
          <>
            <p className="mb-6 text-green-700">{detail}</p>
            <div className="flex flex-col gap-3 sm:flex-row sm:justify-center">
              <Link
                to={PROFILE}
                className="inline-block rounded-lg bg-blue-500 px-6 py-3 font-semibold text-white transition hover:bg-blue-600"
              >
                Личный кабинет
              </Link>
              <Link
                to={SIGN_IN}
                className="inline-block rounded-lg border border-gray-300 px-6 py-3 font-semibold text-gray-800 transition hover:bg-gray-50"
              >
                Вход
              </Link>
            </div>
          </>
        )}
        {phase === "error" && (
          <>
            <p className="mb-6 text-red-600">{detail}</p>
            <Link to={SIGN_IN} className="text-blue-600 underline hover:text-blue-700">
              На страницу входа
            </Link>
          </>
        )}
      </div>
    </div>
  )
}

export default VerifyEmail
