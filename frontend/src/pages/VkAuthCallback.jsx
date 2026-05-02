import React, { useEffect, useState } from "react"
import { useDispatch } from "react-redux"
import { Link, useNavigate } from "react-router-dom"
import { ArrowLeft, Loader2, Scan, Sparkles } from "lucide-react"

import Alert from "../components/ui/Alert"
import { onVerify } from "../asyncActions/onVerify"
import { env } from "../imports/ENV"
import {
  HOME,
  SIGN_IN,
  VK_EXCHANGE,
  VK_LINK,
} from "../imports/ENDPOINTS"
import { publishStoredSessionToOtherTabs } from "../imports/sessionSync"
import {
  consumeVkCodeVerifier,
  formatVkSdkError,
  readVkCallbackParams,
  stashVkLinkSession,
} from "../imports/vkId"

const VkAuthCallback = () => {
  const dispatch = useDispatch()
  const navigate = useNavigate()
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false

    ;(async () => {
      try {
        const params = readVkCallbackParams(
          window.location.search,
          window.location.hash
        )
        if (params.error) {
          if (!cancelled) {
            setError(params.errorDescription || "VK ID вернул ошибку авторизации.")
          }
          return
        }
        if (!params.code || !params.deviceId || !params.state) {
          if (!cancelled) {
            setError("VK ID не вернул обязательные параметры авторизации.")
          }
          return
        }

        const session = consumeVkCodeVerifier(params.state)
        if (!session?.codeVerifier) {
          if (!cancelled) {
            setError("Сессия VK ID истекла. Запустите вход заново.")
          }
          return
        }

        const result = await dispatch(
          onVerify({
            data: {
              code: params.code,
              code_verifier: session.codeVerifier,
              device_id: params.deviceId,
              redirect_uri: env.VK_ID_REDIRECT_URI,
              state: params.state,
              remember_me: session.rememberMe === true,
            },
            endpoint: VK_EXCHANGE,
          })
        ).unwrap()

        if (cancelled) return

        if (result.requiresVkLink && result.vkLinkToken) {
          stashVkLinkSession(result.vkLinkToken, result.userData?.email)
          navigate(VK_LINK, { replace: true })
          return
        }

        if (!result.error && result.userData?.id != null && result.accessToken) {
          publishStoredSessionToOtherTabs()
          navigate(HOME, { replace: true })
          return
        }

        setError(result.error || "Не удалось завершить вход через VK ID.")
      } catch (err) {
        if (!cancelled) {
          setError(`Не удалось завершить вход через VK ID: ${formatVkSdkError(err)}`)
        }
      }
    })()

    return () => {
      cancelled = true
    }
  }, [dispatch, navigate])

  return (
    <div className="flex md:h-[calc(100vh-3.5rem)] md:overflow-hidden items-center justify-center px-4 py-6 md:py-4">
      <div className="w-full max-w-3xl md:max-h-full md:overflow-hidden rounded-2xl shadow-xl ring-1 ring-gray-900/[0.07]">
        <div className="grid lg:grid-cols-[288px_1fr]">
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
                Завершаем безопасный вход через VK ID и подготавливаем рабочую сессию.
              </p>
            </div>

            <div className="relative rounded-2xl border border-white/10 bg-white/[0.07] p-4 backdrop-blur-sm">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-med-300">
                Статус
              </p>
              <p className="mt-2 text-sm leading-relaxed text-white/80">
                {error
                  ? "VK вернул вас в приложение, но последний шаг авторизации потребовал внимания."
                  : "Профиль уже получен. Осталось завершить внутреннюю авторизацию и открыть доступ к системе."}
              </p>
            </div>
          </div>

          <div className="bg-white px-8 py-10">
            <div className="mb-6 flex items-center gap-3 lg:hidden">
              <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl bg-med-600 text-white">
                <Scan size={18} />
              </div>
              <span className="text-sm font-semibold text-gray-700">Skin Cancer AI</span>
            </div>

            <div className="flex items-start justify-between gap-4">
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  {error ? "Вход через VK ID не завершён" : "Завершаем вход"}
                </h1>
                <p className="mt-1 text-sm text-gray-500">
                  {error
                    ? "Проверьте сообщение ниже и попробуйте снова."
                    : "Получаем данные профиля и создаём сессию в Skin Cancer AI."}
                </p>
              </div>
              <div className={`flex h-11 w-11 items-center justify-center rounded-xl ${
                error
                  ? "bg-amber-50 text-amber-600 ring-1 ring-amber-100"
                  : "bg-med-50 text-med-600 ring-1 ring-med-100"
              }`}>
                {error ? <Sparkles size={18} /> : <Loader2 size={18} className="animate-spin" />}
              </div>
            </div>

            {error ? (
              <div className="mt-6 space-y-5">
                <Alert variant="error">
                  {error}
                </Alert>
                <Link
                  to={SIGN_IN}
                  className="inline-flex items-center gap-2 text-sm font-medium text-link"
                >
                  <ArrowLeft size={16} />
                  Вернуться ко входу
                </Link>
              </div>
            ) : (
              <div className="mt-6 space-y-4">
                <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-4">
                  <div className="flex items-start gap-3">
                    <Loader2 size={18} className="mt-0.5 animate-spin text-med-600" />
                    <div>
                      <p className="text-sm font-semibold text-slate-900">
                        Авторизация обрабатывается
                      </p>
                      <p className="mt-1 text-sm leading-relaxed text-slate-600">
                        Обычно этот этап занимает всего пару секунд. Не закрывайте вкладку, пока мы завершаем вход.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="rounded-xl border border-slate-200 bg-white px-4 py-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                      Шаг 1
                    </p>
                    <p className="mt-2 text-sm font-medium text-slate-900">
                      Код от VK получен
                    </p>
                  </div>
                  <div className="rounded-xl border border-slate-200 bg-white px-4 py-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                      Шаг 2
                    </p>
                    <p className="mt-2 text-sm font-medium text-slate-900">
                      Создаём локальную сессию
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default VkAuthCallback
