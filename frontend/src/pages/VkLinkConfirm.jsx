import React, { useEffect, useState } from "react"
import { useDispatch, useSelector } from "react-redux"
import { Link, useNavigate } from "react-router-dom"
import { ArrowLeft, Eye, EyeOff, Link2, ShieldCheck } from "lucide-react"

import Alert from "../components/ui/Alert"
import Button from "../components/ui/Button"
import { onVerify } from "../asyncActions/onVerify"
import { HOME, SIGN_IN, VK_LINK_CONFIRM } from "../imports/ENDPOINTS"
import { clearVkLinkSession, readVkLinkSession } from "../imports/vkId"
import { noError, toggleRememberMe } from "../store/userReducer"
import { publishStoredSessionToOtherTabs } from "../imports/sessionSync"

const VkLinkConfirm = () => {
  const dispatch = useDispatch()
  const navigate = useNavigate()
  const userInfo = useSelector((state) => state.user)
  const [formState, setFormState] = useState({
    email: "",
    password: "",
  })
  const [vkLinkToken, setVkLinkToken] = useState(null)
  const [isPasswordVisible, setIsPasswordVisible] = useState(false)
  const [isRequestPending, setIsRequestPending] = useState(false)
  const [pageError, setPageError] = useState(null)

  useEffect(() => {
    const session = readVkLinkSession()
    if (!session.vkLinkToken) {
      setPageError("Сессия привязки VK ID не найдена или уже истекла.")
      return
    }
    setVkLinkToken(session.vkLinkToken)
    setFormState((prev) => ({
      ...prev,
      email: session.email || prev.email,
    }))
  }, [])

  useEffect(() => {
    if (!isRequestPending && userInfo.error) {
      dispatch(noError())
    }
  }, [formState, isRequestPending, userInfo.error, dispatch])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!vkLinkToken) {
      setPageError("Сессия привязки VK ID не найдена.")
      return
    }
    setIsRequestPending(true)
    try {
      const result = await dispatch(
        onVerify({
          data: {
            vk_link_token: vkLinkToken,
            email: formState.email,
            password: formState.password,
            remember_me: userInfo.isRememberMeChecked,
          },
          endpoint: VK_LINK_CONFIRM,
        })
      ).unwrap()

      if (!result.error && result.userData?.id != null && result.accessToken) {
        clearVkLinkSession()
        publishStoredSessionToOtherTabs()
        navigate(HOME, { replace: true })
      }
    } finally {
      setIsRequestPending(false)
    }
  }

  const canSubmit =
    Boolean(vkLinkToken) &&
    Boolean(formState.email) &&
    Boolean(formState.password) &&
    !isRequestPending

  return (
    <div className="flex min-h-[calc(100vh-3.5rem)] items-center justify-center bg-[radial-gradient(circle_at_top_right,rgba(0,119,255,0.1),transparent_34%),linear-gradient(180deg,#f8fbff_0%,#eef5ff_100%)] px-4 py-10">
      <div className="w-full max-w-xl rounded-[28px] border border-white/80 bg-white/94 p-8 shadow-[0_34px_80px_-34px_rgba(15,23,42,0.35)] ring-1 ring-slate-900/[0.04] backdrop-blur">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-start gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-[#0077FF]/10 text-[#0077FF] ring-1 ring-[#0077FF]/10">
              <Link2 size={20} />
            </div>
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-[#0077FF]">
                VK ID
              </p>
              <h1 className="mt-2 text-2xl font-bold tracking-tight text-slate-950">
                Подтверждение привязки
              </h1>
              <p className="mt-2 max-w-md text-sm leading-7 text-slate-600">
                Такой email уже существует. Подтвердите пароль текущего аккаунта, и мы безопасно привяжем к нему вход через VK ID.
              </p>
            </div>
          </div>
          <div className="hidden h-11 w-11 items-center justify-center rounded-2xl bg-emerald-50 text-emerald-600 ring-1 ring-emerald-100 sm:flex">
            <ShieldCheck size={18} />
          </div>
        </div>

        {pageError && (
          <Alert variant="error" className="mt-6">
            {pageError}
          </Alert>
        )}

        <form className="mt-7 space-y-5" onSubmit={handleSubmit}>
          <div>
            <label htmlFor="vk-link-email" className="input-label">
              Email существующего аккаунта
            </label>
            <input
              id="vk-link-email"
              type="email"
              autoComplete="email"
              className="input-field"
              value={formState.email}
              onChange={(e) =>
                setFormState((prev) => ({ ...prev, email: e.target.value }))
              }
            />
          </div>

          <div>
            <label htmlFor="vk-link-password" className="input-label">
              Пароль
            </label>
            <div className="relative">
              <input
                id="vk-link-password"
                type={isPasswordVisible ? "text" : "password"}
                autoComplete="current-password"
                className="input-field pr-10"
                value={formState.password}
                onChange={(e) =>
                  setFormState((prev) => ({ ...prev, password: e.target.value }))
                }
              />
              <button
                type="button"
                onClick={() => setIsPasswordVisible((v) => !v)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
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
            <Alert variant="error">
              {userInfo.error}
            </Alert>
          )}

          <Button type="submit" disabled={!canSubmit} className="w-full">
            {isRequestPending ? "Привязываем VK ID..." : "Подтвердить и войти"}
          </Button>
        </form>

        <Link
          to={SIGN_IN}
          className="mt-6 inline-flex items-center gap-2 text-sm font-semibold text-link"
        >
          <ArrowLeft size={16} />
          Вернуться к обычному входу
        </Link>
      </div>
    </div>
  )
}

export default VkLinkConfirm
