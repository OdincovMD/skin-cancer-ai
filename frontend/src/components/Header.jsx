import React, { useEffect, useMemo, useState } from "react"
import { useDispatch, useSelector } from "react-redux"
import { Link, useLocation } from "react-router-dom"
import {
  AlertCircle,
  CheckCircle2,
  Loader2,
  LogIn,
  Menu,
  Scan,
  User,
} from "lucide-react"

import { useAvatarObjectUrl } from "../hooks/useAvatarObjectUrl"
import { env } from "../imports/ENV"
import { SIGN_IN, PROFILE } from "../imports/ENDPOINTS"
import { noError } from "../store/userReducer"

const Header = ({ toggleSidebar }) => {
  const dispatch = useDispatch()
  const userInfo = useSelector((state) => state.user)
  const { pathname } = useLocation()
  const [serviceHealth, setServiceHealth] = useState({
    status: "loading",
    healthyCount: 0,
    totalCount: 0,
    services: {},
  })

  const firstName = userInfo.userData?.firstName || ""
  const lastName = userInfo.userData?.lastName || ""
  const email = userInfo.userData?.email || ""
  const displayName = `${firstName} ${lastName}`.trim() || email
  const initials =
    (firstName.charAt(0) || "").toUpperCase() +
    (userInfo.userData?.lastName?.charAt(0) || "").toUpperCase() ||
    email.charAt(0).toUpperCase()

  const avatarUrl = useAvatarObjectUrl(
    userInfo.accessToken,
    userInfo.avatarRevision
  )
  const mlService = serviceHealth.services?.ml || null
  const mlModels =
    mlService?.models && typeof mlService.models === "object"
      ? mlService.models
      : null

  useEffect(() => {
    let isCancelled = false

    const fetchHealth = async () => {
      try {
        const base = env.BACKEND_URL.replace(/\/$/, "")
        const response = await fetch(`${base}/health`, {
          headers: { accept: "application/json" },
        })
        const payload = await response.json().catch(() => ({}))
        if (isCancelled) return
        const services =
          payload?.services && typeof payload.services === "object"
            ? payload.services
            : {}
        setServiceHealth({
          status:
            typeof payload?.status === "string"
              ? payload.status
              : response.ok
                ? "ok"
                : "unavailable",
          healthyCount:
            typeof payload?.healthy_count === "number"
              ? payload.healthy_count
              : 0,
          totalCount:
            typeof payload?.total_count === "number"
              ? payload.total_count
              : Object.keys(services).length,
          services,
        })
      } catch {
        if (isCancelled) return
        setServiceHealth({
          status: "unavailable",
          healthyCount: 0,
          totalCount: 0,
          services: {},
        })
      }
    }

    fetchHealth()
    const intervalId = window.setInterval(fetchHealth, 30000)

    return () => {
      isCancelled = true
      window.clearInterval(intervalId)
    }
  }, [])

  const serviceBadge = useMemo(() => {
    if (serviceHealth.status === "loading") {
      return {
        label: "Проверка ML...",
        className:
          "border-slate-200 bg-slate-50 text-slate-600",
        Icon: Loader2,
        iconClassName: "animate-spin",
      }
    }

    if (!mlService?.ok) {
      return {
        label: "ML недоступен",
        className:
          "border-rose-200 bg-rose-50 text-rose-700",
        Icon: AlertCircle,
        iconClassName: "",
      }
    }

    const loadedCount =
      typeof mlModels?.loaded_count === "number" ? mlModels.loaded_count : null
    const totalCount =
      typeof mlModels?.total_count === "number" ? mlModels.total_count : null
    const hasModelProgress = loadedCount !== null && totalCount !== null && totalCount > 0

    if (!hasModelProgress) {
      return {
        label: "ML доступен",
        className:
          "border-emerald-200 bg-emerald-50 text-emerald-700",
        Icon: CheckCircle2,
        iconClassName: "",
      }
    }

    if (loadedCount === totalCount) {
      return {
        label: "ML готов",
        className:
          "border-emerald-200 bg-emerald-50 text-emerald-700",
        Icon: CheckCircle2,
        iconClassName: "",
      }
    }

    return {
      label: "ML подготавливается",
      className:
        "border-amber-200 bg-amber-50 text-amber-700",
      Icon: AlertCircle,
      iconClassName: "",
    }
  }, [mlModels, mlService, serviceHealth.status])

  const tooltipLines = useMemo(() => {
    if (!mlService?.ok) {
      return [
        "ML-сервис сейчас недоступен.",
      ]
    }

    const loadedCount =
      typeof mlModels?.loaded_count === "number" ? mlModels.loaded_count : null
    const totalCount =
      typeof mlModels?.total_count === "number" ? mlModels.total_count : null
    const hasModelProgress = loadedCount !== null && totalCount !== null && totalCount > 0

    if (!hasModelProgress) {
      return [
        "ML-сервис доступен и готов принимать запросы.",
      ]
    }

    if (loadedCount === totalCount) {
      return [
        "Основные ML-модели уже загружены в память.",
        "Обычно это значит, что дополнительной задержки на первом запросе не будет.",
      ]
    }

    return [
      "Часть ML-моделей ещё загружается.",
      "Из-за этого первый запрос в некоторых сценариях может выполняться чуть дольше.",
    ]
  }, [mlModels, mlService])

  return (
    <header className="fixed left-0 right-0 top-0 z-50 h-14 border-b border-slate-200 bg-white/90 shadow-sm backdrop-blur-xl">
      <div className="mx-auto flex h-full max-w-screen-2xl items-center px-3 sm:px-4">
        <button
          onClick={toggleSidebar}
          className="-ml-1 flex h-10 w-10 items-center justify-center rounded-lg border border-transparent text-slate-500 transition-colors hover:border-slate-200 hover:bg-slate-50 hover:text-slate-900"
          aria-label="Меню"
        >
          <Menu size={20} />
        </button>

        <Link to="/" className="ml-3 flex items-center gap-3 select-none">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-med-600 text-white shadow-sm">
            <Scan size={18} strokeWidth={2.2} />
          </div>
          <div className="leading-tight">
            <span className="block text-sm font-bold tracking-tight text-slate-950 sm:text-base">
              Skin Cancer AI
            </span>
            <span className="hidden text-[11px] font-medium text-slate-400 sm:block">
              Дерматоскопическая диагностика
            </span>
          </div>
        </Link>

        <div
          className="ml-auto flex items-center gap-2 sm:gap-3"
          onClick={() => {
            if (pathname !== SIGN_IN) dispatch(noError())
          }}
        >
          <div className="group relative hidden md:flex">
            <div
              className={`items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${serviceBadge.className} flex`}
            >
              <serviceBadge.Icon
                size={14}
                className={serviceBadge.iconClassName}
              />
              {serviceBadge.label}
            </div>
            <div className="pointer-events-none absolute right-0 top-[calc(100%+0.6rem)] z-20 w-72 rounded-xl border border-slate-200 bg-white p-3 text-left text-[11px] leading-5 text-slate-600 opacity-0 shadow-xl transition-all duration-150 group-hover:translate-y-0 group-hover:opacity-100 group-focus-within:translate-y-0 group-focus-within:opacity-100">
              {tooltipLines.map((line) => (
                <p key={line}>{line}</p>
              ))}
            </div>
          </div>

          {userInfo.userData?.id && userInfo.accessToken ? (
            <Link
              to={PROFILE}
              className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white py-1 pl-1 pr-2 shadow-sm transition-colors hover:bg-slate-50 sm:pr-3"
            >
              {avatarUrl ? (
                <img
                  src={avatarUrl}
                  alt=""
                  className="h-8 w-8 shrink-0 rounded-md object-cover ring-1 ring-slate-200"
                />
              ) : (
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-teal-50 text-xs font-semibold text-med-700 ring-1 ring-teal-100">
                  {initials || <User size={16} />}
                </div>
              )}
              <div className="hidden min-w-0 leading-tight sm:block">
                <span className="block max-w-36 truncate text-sm font-semibold text-slate-800">
                  {displayName || "Профиль"}
                </span>
                <span className="block text-[11px] text-slate-400">
                  Личный кабинет
                </span>
              </div>
            </Link>
          ) : (
            <Link
              to={SIGN_IN}
              className="flex items-center gap-2 rounded-lg bg-med-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-med-700"
            >
              <LogIn size={16} />
              <span className="hidden sm:inline">Войти</span>
            </Link>
          )}
        </div>
      </div>
    </header>
  )
}

export default Header
