import React, { useCallback, useEffect, useState } from "react"
import { useDispatch, useSelector } from "react-redux"
import { Link } from "react-router-dom"
import {
  Activity,
  BadgeCheck,
  Braces,
  Camera,
  Save,
  Lock,
  Mail,
  Clock,
  FileText,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  CheckCircle2,
  ClipboardCheck,
  Database,
  Fingerprint,
  Globe2,
  Loader2,
  User,
  UserRoundCog,
  ImageIcon,
  RefreshCw,
  Key,
  KeyRound,
  Copy,
  Trash2,
  Server,
  ShieldCheck,
  SquareTerminal,
  Workflow,
} from "lucide-react"

import { useAvatarObjectUrl } from "../hooks/useAvatarObjectUrl"
import { handleHistoryRequest } from "../asyncActions/handleHistoryRequest"
import { env } from "../imports/ENV"
import { bearerAuthHeaders } from "../imports/authHeaders"
import {
  API_DOCS,
  CHANGE_PASSWORD,
  HISTORY_IMAGE,
  ME_API_TOKEN,
  ME_API_TOKEN_ROTATE,
  ME_AVATAR,
  ME_PROFILE,
  RESEND_VERIFICATION_EMAIL,
} from "../imports/ENDPOINTS"
import { getValues } from "../imports/HELPERS"
import TreeComponent from "../components/Tree"
import BucketLabelsDisclosure, {
  formatFeatureLabelText,
} from "../components/ui/BucketLabelsDisclosure"
import Button from "../components/ui/Button"
import {
  bumpAvatarRevision,
  mergeUserData,
  setVerificationResendCooldownFromSeconds,
} from "../store/userReducer"

const SectionHeader = ({ icon: Icon, title, eyebrow, className = "" }) => (
  <div className={`mb-5 flex items-start gap-3 ${className}`.trim()}>
    <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-teal-100 bg-teal-50 text-med-700">
      <Icon size={18} aria-hidden />
    </div>
    <div>
      {eyebrow && (
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
          {eyebrow}
        </p>
      )}
      <h2 className="text-base font-semibold text-slate-950">{title}</h2>
    </div>
  </div>
)

const StatusBadge = ({ tone = "slate", icon: Icon, children }) => {
  const tones = {
    green: "border-emerald-200 bg-emerald-50 text-emerald-700",
    amber: "border-amber-200 bg-amber-50 text-amber-700",
    red: "border-red-200 bg-red-50 text-red-700",
    teal: "border-teal-200 bg-teal-50 text-teal-700",
    slate: "border-slate-200 bg-slate-50 text-slate-600",
  }

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold ${
        tones[tone] || tones.slate
      }`}
    >
      {Icon && <Icon size={13} aria-hidden />}
      {children}
    </span>
  )
}

const Profile = () => {
  const dispatch = useDispatch()
  const userInfo = useSelector((state) => state.user)

  const [history, setHistory] = useState([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [openHistoryImageKey, setOpenHistoryImageKey] = useState(null)
  const [openTreeKey, setOpenTreeKey] = useState(null)
  const [openDescriptionKey, setOpenDescriptionKey] = useState(null)
  const [resendPending, setResendPending] = useState(false)
  const [resendHint, setResendHint] = useState("")
  const [, tickResendCooldown] = useState(0)
  const verificationResendUntilMs = userInfo.verificationResendUntilMs

  useEffect(() => {
    if (!verificationResendUntilMs || Date.now() >= verificationResendUntilMs) return
    const id = window.setInterval(() => tickResendCooldown((n) => n + 1), 1000)
    return () => window.clearInterval(id)
  }, [verificationResendUntilMs])

  const resendCooldownRemaining =
    verificationResendUntilMs && verificationResendUntilMs > Date.now()
      ? Math.ceil((verificationResendUntilMs - Date.now()) / 1000)
      : 0

  const [pwdCurrent, setPwdCurrent] = useState("")
  const [pwdNew, setPwdNew] = useState("")
  const [pwdConfirm, setPwdConfirm] = useState("")
  const [pwdPending, setPwdPending] = useState(false)
  const [pwdMessage, setPwdMessage] = useState({ type: "", text: "" })

  const pwdNewValid = Boolean(pwdNew.match(/^[0-9A-Za-z]{8,}$/))
  const pwdMatch = pwdNew === pwdConfirm && pwdConfirm.length > 0

  const avatarDisplayUrl = useAvatarObjectUrl(
    userInfo.accessToken,
    userInfo.avatarRevision
  )
  const [editFirstName, setEditFirstName] = useState("")
  const [editLastName, setEditLastName] = useState("")
  const [profilePending, setProfilePending] = useState(false)
  const [profileMessage, setProfileMessage] = useState({ type: "", text: "" })
  const [avatarUploadPending, setAvatarUploadPending] = useState(false)
  const [avatarMessage, setAvatarMessage] = useState({ type: "", text: "" })

  const [apiTokStatus, setApiTokStatus] = useState(null)
  const [apiTokPlain, setApiTokPlain] = useState(null)
  const [apiTokLoading, setApiTokLoading] = useState(false)
  const [apiTokActionPending, setApiTokActionPending] = useState(false)
  const [apiTokMsg, setApiTokMsg] = useState({ type: "", text: "" })

  const loadApiTokenStatus = useCallback(async () => {
    if (!userInfo.accessToken || !userInfo.emailVerified) return
    setApiTokLoading(true)
    setApiTokMsg({ type: "", text: "" })
    try {
      const base = env.BACKEND_URL.replace(/\/$/, "")
      const res = await fetch(`${base}${ME_API_TOKEN}`, {
        headers: {
          accept: "application/json",
          ...bearerAuthHeaders(userInfo.accessToken),
        },
      })
      const data = await res.json().catch(() => ({}))
      if (res.status === 401) {
        setApiTokMsg({ type: "err", text: "Сессия истекла." })
        return
      }
      if (!res.ok) {
        const d = data.detail
        setApiTokMsg({
          type: "err",
          text:
            typeof d === "string"
              ? d
              : "Не удалось загрузить статус API-ключа.",
        })
        return
      }
      setApiTokStatus({
        has_token: Boolean(data.has_token),
        created_at: data.created_at,
        display_label: data.display_label,
      })
    } catch (e) {
      setApiTokMsg({ type: "err", text: String(e?.message || e) })
    } finally {
      setApiTokLoading(false)
    }
  }, [userInfo.accessToken, userInfo.emailVerified])

  useEffect(() => {
    loadApiTokenStatus()
  }, [loadApiTokenStatus])

  useEffect(() => {
    setEditFirstName(userInfo.userData?.firstName ?? "")
    setEditLastName(userInfo.userData?.lastName ?? "")
  }, [userInfo.userData?.id, userInfo.userData?.firstName, userInfo.userData?.lastName])

  const handleAvatarFile = async (event) => {
    const file = event.target.files?.[0]
    event.target.value = ""
    setAvatarMessage({ type: "", text: "" })
    if (!file || !userInfo.accessToken) return
    setAvatarUploadPending(true)
    try {
      const base = env.BACKEND_URL.replace(/\/$/, "")
      const fd = new FormData()
      fd.append("file", file)
      const res = await fetch(`${base}${ME_AVATAR}`, {
        method: "POST",
        headers: { ...bearerAuthHeaders(userInfo.accessToken) },
        body: fd,
      })
      const data = await res.json().catch(() => ({}))
      if (res.status === 401) {
        setAvatarMessage({ type: "err", text: "Сессия истекла. Войдите снова." })
        return
      }
      if (data.error) {
        setAvatarMessage({ type: "err", text: data.error })
        return
      }
      setAvatarMessage({ type: "ok", text: "Фото обновлено" })
      dispatch(bumpAvatarRevision())
    } catch (err) {
      setAvatarMessage({ type: "err", text: String(err?.message || err) })
    } finally {
      setAvatarUploadPending(false)
    }
  }

  const submitProfileNames = async (e) => {
    e.preventDefault()
    setProfileMessage({ type: "", text: "" })
    const fn = editFirstName.trim()
    const ln = editLastName.trim()
    const body = {}
    if (fn) body.firstName = fn
    if (ln) body.lastName = ln
    if (Object.keys(body).length === 0) {
      setProfileMessage({ type: "err", text: "Укажите имя или фамилию." })
      return
    }
    setProfilePending(true)
    try {
      const base = env.BACKEND_URL.replace(/\/$/, "")
      const res = await fetch(`${base}${ME_PROFILE}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          accept: "application/json",
          ...bearerAuthHeaders(userInfo.accessToken),
        },
        body: JSON.stringify(body),
      })
      const data = await res.json().catch(() => ({}))
      if (res.status === 401) {
        setProfileMessage({ type: "err", text: "Сессия истекла." })
        return
      }
      if (res.status === 422) {
        const d = data.detail
        const msg = Array.isArray(d) && d[0]?.msg != null ? String(d[0].msg) : "Проверьте данные."
        setProfileMessage({ type: "err", text: msg })
        return
      }
      if (data.error) {
        setProfileMessage({ type: "err", text: data.error })
        return
      }
      if (data.userData) dispatch(mergeUserData(data.userData))
      setProfileMessage({ type: "ok", text: "Сохранено" })
    } catch (err) {
      setProfileMessage({ type: "err", text: String(err?.message || err) })
    } finally {
      setProfilePending(false)
    }
  }

  const submitChangePassword = async (e) => {
    e.preventDefault()
    setPwdMessage({ type: "", text: "" })
    if (!pwdNewValid) {
      setPwdMessage({ type: "err", text: "Не менее 8 символов, латиница и цифры." })
      return
    }
    if (!pwdMatch) {
      setPwdMessage({ type: "err", text: "Пароли не совпадают." })
      return
    }
    setPwdPending(true)
    try {
      const base = env.BACKEND_URL.replace(/\/$/, "")
      const res = await fetch(`${base}${CHANGE_PASSWORD}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          accept: "application/json",
          ...bearerAuthHeaders(userInfo.accessToken),
        },
        body: JSON.stringify({ current_password: pwdCurrent, new_password: pwdNew }),
      })
      const data = await res.json().catch(() => ({}))
      if (res.status === 401) {
        setPwdMessage({ type: "err", text: "Сессия истекла." })
        return
      }
      if (res.status === 422) {
        const d = data.detail
        const msg = Array.isArray(d) && d[0]?.msg != null ? String(d[0].msg) : "Проверьте данные."
        setPwdMessage({ type: "err", text: msg })
        return
      }
      if (data.error) {
        setPwdMessage({ type: "err", text: data.error })
        return
      }
      setPwdCurrent("")
      setPwdNew("")
      setPwdConfirm("")
      setPwdMessage({ type: "ok", text: "Пароль изменён" })
    } catch (err) {
      setPwdMessage({ type: "err", text: String(err?.message || err) })
    } finally {
      setPwdPending(false)
    }
  }

  const resendVerification = async () => {
    setResendHint("")
    setResendPending(true)
    try {
      const base = env.BACKEND_URL.replace(/\/$/, "")
      const res = await fetch(`${base}${RESEND_VERIFICATION_EMAIL}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          accept: "application/json",
          ...bearerAuthHeaders(userInfo.accessToken),
        },
        body: "{}",
      })
      const data = await res.json().catch(() => ({}))
      if (data.error) {
        if (data.retry_after_seconds != null)
          dispatch(setVerificationResendCooldownFromSeconds(data.retry_after_seconds))
        setResendHint(data.error)
      } else {
        setResendHint("Письмо отправлено. Проверьте почту.")
        dispatch(
          setVerificationResendCooldownFromSeconds(data.verification_resend_after_seconds ?? 120)
        )
      }
    } catch (e) {
      setResendHint(String(e.message || e))
    } finally {
      setResendPending(false)
    }
  }

  const detailMessage = (data) => {
    const d = data?.detail
    if (typeof d === "string") return d
    if (Array.isArray(d) && d[0]?.msg != null) return String(d[0].msg)
    return null
  }

  const issueApiToken = async () => {
    setApiTokActionPending(true)
    setApiTokMsg({ type: "", text: "" })
    setApiTokPlain(null)
    try {
      const base = env.BACKEND_URL.replace(/\/$/, "")
      const res = await fetch(`${base}${ME_API_TOKEN}`, {
        method: "POST",
        headers: {
          accept: "application/json",
          ...bearerAuthHeaders(userInfo.accessToken),
        },
      })
      const data = await res.json().catch(() => ({}))
      if (res.status === 401) {
        setApiTokMsg({ type: "err", text: "Сессия истекла." })
        return
      }
      if (res.status === 409) {
        setApiTokMsg({
          type: "err",
          text: detailMessage(data) || "Ключ уже выпущен. Используйте перевыпуск.",
        })
        await loadApiTokenStatus()
        return
      }
      if (!res.ok) {
        setApiTokMsg({
          type: "err",
          text: detailMessage(data) || "Не удалось выпустить ключ.",
        })
        return
      }
      setApiTokPlain(data.token ?? "")
      setApiTokStatus({
        has_token: true,
        created_at: data.created_at,
        display_label: data.display_label ?? "scai_••••••••",
      })
      setApiTokMsg({
        type: "ok",
        text: "Сохраните ключ в надёжном месте — полное значение показывается только сейчас.",
      })
    } catch (e) {
      setApiTokMsg({ type: "err", text: String(e?.message || e) })
    } finally {
      setApiTokActionPending(false)
    }
  }

  const rotateApiToken = async () => {
    setApiTokActionPending(true)
    setApiTokMsg({ type: "", text: "" })
    setApiTokPlain(null)
    try {
      const base = env.BACKEND_URL.replace(/\/$/, "")
      const res = await fetch(`${base}${ME_API_TOKEN_ROTATE}`, {
        method: "POST",
        headers: {
          accept: "application/json",
          ...bearerAuthHeaders(userInfo.accessToken),
        },
      })
      const data = await res.json().catch(() => ({}))
      if (res.status === 401) {
        setApiTokMsg({ type: "err", text: "Сессия истекла." })
        return
      }
      if (res.status === 409) {
        setApiTokMsg({
          type: "err",
          text: detailMessage(data) || "Сначала выпустите ключ.",
        })
        return
      }
      if (!res.ok) {
        setApiTokMsg({
          type: "err",
          text: detailMessage(data) || "Не удалось перевыпустить ключ.",
        })
        return
      }
      setApiTokPlain(data.token ?? "")
      setApiTokStatus({
        has_token: true,
        created_at: data.created_at,
        display_label: data.display_label ?? "scai_••••••••",
      })
      setApiTokMsg({
        type: "ok",
        text: "Новый ключ показан один раз. Старый перестал действовать.",
      })
    } catch (e) {
      setApiTokMsg({ type: "err", text: String(e?.message || e) })
    } finally {
      setApiTokActionPending(false)
    }
  }

  const revokeApiToken = async () => {
    if (!window.confirm("Отозвать API-ключ? Интеграции перестанут работать.")) return
    setApiTokActionPending(true)
    setApiTokMsg({ type: "", text: "" })
    try {
      const base = env.BACKEND_URL.replace(/\/$/, "")
      const res = await fetch(`${base}${ME_API_TOKEN}`, {
        method: "DELETE",
        headers: {
          accept: "application/json",
          ...bearerAuthHeaders(userInfo.accessToken),
        },
      })
      const data = await res.json().catch(() => ({}))
      if (res.status === 401) {
        setApiTokMsg({ type: "err", text: "Сессия истекла." })
        return
      }
      if (!res.ok) {
        setApiTokMsg({
          type: "err",
          text: detailMessage(data) || "Не удалось отозвать ключ.",
        })
        return
      }
      setApiTokPlain(null)
      setApiTokStatus({ has_token: false, created_at: null, display_label: null })
      setApiTokMsg({ type: "ok", text: "Ключ отозван." })
    } catch (e) {
      setApiTokMsg({ type: "err", text: String(e?.message || e) })
    } finally {
      setApiTokActionPending(false)
    }
  }

  const copyApiToken = async () => {
    if (!apiTokPlain) return
    try {
      await navigator.clipboard.writeText(apiTokPlain)
      setApiTokMsg({ type: "ok", text: "Скопировано в буфер обмена." })
    } catch {
      setApiTokMsg({ type: "err", text: "Не удалось скопировать. Выделите и скопируйте вручную." })
    }
  }

  const fetchHistory = () => {
    setHistoryLoading(true)
    handleHistoryRequest(userInfo.accessToken).then((response) => {
      setOpenHistoryImageKey(null)
      setOpenTreeKey(null)
      setHistory(Array.isArray(response) ? response : [])
      setHistoryLoading(false)
    })
  }

  const initials =
    ((userInfo.userData?.firstName || "").charAt(0) +
      (userInfo.userData?.lastName || "").charAt(0)).toUpperCase()
  const displayName =
    `${userInfo.userData?.firstName ?? ""} ${
      userInfo.userData?.lastName ?? ""
    }`.trim() || "Пользователь"
  const apiTokenLabel = apiTokStatus?.display_label || "scai_••••••••"

  const parseResult = (raw) => {
    if (raw == null || raw === "") return {}
    try {
      const parsed = typeof raw === "string" ? JSON.parse(raw) : raw
      return parsed != null && typeof parsed === "object" && !Array.isArray(parsed)
        ? parsed
        : {}
    } catch {
      return {}
    }
  }

  const formatDate = (raw) => {
    const m = /^(?<date>.*)T(?<time>.*)\..*\+/.exec(raw)
    return m ? `${m.groups.date} ${m.groups.time}` : raw
  }

  const apiCreatedAt = apiTokStatus?.created_at
    ? formatDate(apiTokStatus.created_at)
    : null

  const extractFileName = (raw) => {
    const m = /^(?:.*?_){3}(?<filename>.*)$/.exec(raw)
    return m?.groups?.filename ?? raw
  }

  return (
    <div className="space-y-6">
      {userInfo.accessToken && !userInfo.emailVerified && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 px-5 py-4">
          <div className="flex items-start gap-3">
            <Mail size={20} className="mt-0.5 flex-shrink-0 text-amber-600" />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold text-amber-900">
                Адрес email не подтверждён
              </p>
              <p className="mt-1 text-sm leading-relaxed text-amber-800">
                Загрузка изображений, история и API-ключ станут доступны после
                подтверждения почты.
              </p>
              <div className="mt-3 flex flex-wrap items-center gap-3">
                <Button
                  type="button"
                  onClick={resendVerification}
                  disabled={resendPending || resendCooldownRemaining > 0}
                  className="bg-amber-700 text-xs hover:bg-amber-800 focus-visible:ring-amber-500"
                >
                  {resendPending
                    ? "Отправка..."
                    : resendCooldownRemaining > 0
                      ? `Повтор через ${resendCooldownRemaining} с`
                      : "Отправить письмо"}
                </Button>
                {resendHint && (
                  <p
                    className={`text-xs ${
                      resendHint.startsWith("Письмо")
                        ? "text-emerald-700"
                        : "text-amber-900"
                    }`}
                  >
                    {resendHint}
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
        <div className="flex flex-col gap-5 p-6 sm:flex-row sm:items-center">
          <div className="relative group h-24 w-24 flex-shrink-0">
            <div className="flex h-24 w-24 items-center justify-center overflow-hidden rounded-xl bg-gradient-to-br from-teal-100 via-cyan-50 to-white ring-1 ring-teal-200">
              {avatarDisplayUrl ? (
                <img
                  src={avatarDisplayUrl}
                  alt="Аватар"
                  className="h-full w-full object-cover"
                />
              ) : (
                <span className="text-2xl font-bold text-med-700">
                  {initials || <User size={32} />}
                </span>
              )}
            </div>
            <label className="absolute inset-0 flex cursor-pointer items-center justify-center rounded-xl bg-slate-950/45 opacity-0 transition-opacity group-hover:opacity-100">
              <Camera size={20} className="text-white" />
              <input
                type="file"
                accept="image/jpeg,image/png,image/webp"
                className="hidden"
                disabled={avatarUploadPending}
                onChange={handleAvatarFile}
              />
            </label>
          </div>

          <div className="min-w-0 flex-1">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              Профиль пользователя
            </p>
            <h1 className="mt-1 truncate text-2xl font-bold text-slate-950">
              {displayName}
            </h1>
            <p className="mt-1 truncate text-sm text-slate-500">
              {userInfo.userData?.email}
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              {userInfo.emailVerified ? (
                <StatusBadge tone="green" icon={BadgeCheck}>
                  Email подтверждён
                </StatusBadge>
              ) : (
                <StatusBadge tone="amber" icon={AlertTriangle}>
                  Требуется подтверждение
                </StatusBadge>
              )}
              <StatusBadge
                tone={apiTokStatus?.has_token ? "teal" : "slate"}
                icon={KeyRound}
              >
                {apiTokStatus?.has_token ? "API включён" : "API не выпущен"}
              </StatusBadge>
            </div>
            {avatarMessage.text && (
              <p
                className={`mt-3 text-xs ${
                  avatarMessage.type === "ok"
                    ? "text-emerald-700"
                    : "text-red-600"
                }`}
              >
                {avatarMessage.text}
              </p>
            )}
          </div>
        </div>
      </section>

      {userInfo.emailVerified && (
        <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
          <div className="grid gap-0 lg:grid-cols-[minmax(0,1fr)_360px]">
            <div className="p-6">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                <SectionHeader
                  icon={SquareTerminal}
                  title="Доступ по API"
                  eyebrow="Developer access"
                />
                <StatusBadge
                  tone={apiTokStatus?.has_token ? "green" : "slate"}
                  icon={apiTokStatus?.has_token ? ShieldCheck : Fingerprint}
                >
                  {apiTokStatus?.has_token ? "Ключ активен" : "Ключ не выпущен"}
                </StatusBadge>
              </div>

              <p className="max-w-2xl text-sm leading-7 text-slate-600">
                API-ключ открывает доступ к классификации снимков через{" "}
                <Link to={API_DOCS} className="text-link">
                  HTTP API v1
                </Link>
                . Передавайте его в заголовке{" "}
                <code className="rounded bg-slate-100 px-1.5 py-0.5 text-xs text-slate-700">
                  X-API-Key
                </code>
                , а лимит сервера уже защищает интеграции от перегрузки.
              </p>

              <div className="mt-5 grid gap-3 sm:grid-cols-3">
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                  <Server size={18} className="text-med-700" />
                  <p className="mt-3 text-sm font-semibold text-slate-950">
                    Endpoint
                  </p>
                  <p className="mt-1 text-xs text-slate-500">/api/v1/uploadfile</p>
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                  <Fingerprint size={18} className="text-med-700" />
                  <p className="mt-3 text-sm font-semibold text-slate-950">
                    Авторизация
                  </p>
                  <p className="mt-1 text-xs text-slate-500">X-API-Key</p>
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                  <Activity size={18} className="text-med-700" />
                  <p className="mt-3 text-sm font-semibold text-slate-950">
                    Ограничение
                  </p>
                  <p className="mt-1 text-xs text-slate-500">5 запросов/мин</p>
                </div>
              </div>

              <div className="mt-5 rounded-lg border border-med-200 bg-med-50 p-4 text-med-950">
                <div className="flex items-center justify-between gap-3">
                  <div className="min-w-0">
                    <p className="text-xs font-semibold uppercase tracking-wide text-med-700">
                      Token
                    </p>
                    <p className="mt-1 truncate font-mono text-sm">
                      {apiTokStatus?.has_token ? apiTokenLabel : "scai_••••••••"}
                    </p>
                  </div>
                  {apiCreatedAt && (
                    <span className="rounded-full border border-med-200 bg-white px-3 py-1 text-xs text-med-700">
                      {apiCreatedAt}
                    </span>
                  )}
                </div>

                {apiTokPlain && (
                  <div className="mt-4">
                    <label className="text-xs font-semibold uppercase tracking-wide text-med-700">
                      Полный ключ показывается один раз
                    </label>
                    <textarea
                      readOnly
                      className="mt-2 min-h-[5rem] w-full rounded-lg border border-med-200 bg-white px-3 py-3 font-mono text-xs text-med-950 outline-none"
                      value={apiTokPlain}
                    />
                    <Button
                      type="button"
                      variant="secondary"
                      className="mt-3 border-med-200 bg-white text-xs text-med-700 hover:bg-med-100"
                      onClick={copyApiToken}
                    >
                      <Copy size={14} /> Копировать ключ
                    </Button>
                  </div>
                )}
              </div>

              {apiTokLoading && !apiTokStatus ? (
                <p className="mt-4 flex items-center gap-2 text-sm text-slate-500">
                  <Loader2 size={16} className="animate-spin" /> Загрузка статуса...
                </p>
              ) : (
                <div className="mt-5 flex flex-wrap gap-2">
                  {!apiTokStatus?.has_token ? (
                    <Button
                      type="button"
                      disabled={apiTokActionPending}
                      onClick={issueApiToken}
                    >
                      {apiTokActionPending ? (
                        <>
                          <Loader2 size={16} className="animate-spin" /> Выпуск...
                        </>
                      ) : (
                        <>
                          <Key size={16} /> Выпустить ключ
                        </>
                      )}
                    </Button>
                  ) : (
                    <>
                      <Button
                        type="button"
                        variant="secondary"
                        disabled={apiTokActionPending}
                        onClick={rotateApiToken}
                      >
                        {apiTokActionPending ? (
                          <>
                            <Loader2 size={16} className="animate-spin" /> Обновление...
                          </>
                        ) : (
                          <>
                            <RefreshCw size={16} /> Перевыпустить
                          </>
                        )}
                      </Button>
                      <Button
                        type="button"
                        variant="secondary"
                        className="border-red-200 text-red-700 hover:bg-red-50"
                        disabled={apiTokActionPending}
                        onClick={revokeApiToken}
                      >
                        <Trash2 size={16} /> Отозвать
                      </Button>
                    </>
                  )}
                  <Button type="button" variant="secondary" to={API_DOCS}>
                    <Braces size={16} /> Документация
                  </Button>
                </div>
              )}

              {apiTokMsg.text && (
                <p
                  className={`mt-4 rounded-lg border px-4 py-3 text-sm ${
                    apiTokMsg.type === "ok"
                      ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                      : "border-red-200 bg-red-50 text-red-700"
                  }`}
                >
                  {apiTokMsg.text}
                </p>
              )}
            </div>

            <div className="border-t border-slate-200 bg-slate-50 p-6 lg:border-l lg:border-t-0">
              <img
                src="/images/api-console.svg"
                alt="Схема API-интеграции"
                className="h-auto w-full rounded-lg border border-slate-200 bg-white"
              />
              <div className="mt-4 space-y-3">
                <div className="flex items-start gap-3">
                  <Workflow size={18} className="mt-0.5 text-med-700" />
                  <p className="text-sm leading-relaxed text-slate-600">
                    Загрузка снимка, постановка задачи и получение результата
                    остаются в одном API-потоке.
                  </p>
                </div>
                <div className="flex items-start gap-3">
                  <Database size={18} className="mt-0.5 text-med-700" />
                  <p className="text-sm leading-relaxed text-slate-600">
                    История и изображения доступны только владельцу ключа.
                  </p>
                </div>
                <div className="flex items-start gap-3">
                  <Globe2 size={18} className="mt-0.5 text-med-700" />
                  <p className="text-sm leading-relaxed text-slate-600">
                    Подходит для личных кабинетов, внутренних панелей и
                    исследовательских инструментов.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
        <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <SectionHeader
            icon={UserRoundCog}
            title="Личные данные"
            eyebrow="Account"
          />
          <form onSubmit={submitProfileNames} className="space-y-4">
            <div>
              <label htmlFor="prof-fn" className="input-label">Имя</label>
              <input
                id="prof-fn"
                type="text"
                autoComplete="given-name"
                value={editFirstName}
                onChange={(e) => setEditFirstName(e.target.value)}
                maxLength={100}
                className="input-field"
              />
            </div>
            <div>
              <label htmlFor="prof-ln" className="input-label">Фамилия</label>
              <input
                id="prof-ln"
                type="text"
                autoComplete="family-name"
                value={editLastName}
                onChange={(e) => setEditLastName(e.target.value)}
                maxLength={100}
                className="input-field"
              />
            </div>
            {profileMessage.text && (
              <p
                className={`text-sm ${
                  profileMessage.type === "ok"
                    ? "text-emerald-700"
                    : "text-red-600"
                }`}
              >
                {profileMessage.text}
              </p>
            )}
            <Button type="submit" disabled={profilePending}>
              {profilePending ? (
                <>
                  <Loader2 size={16} className="animate-spin" /> Сохранение...
                </>
              ) : (
                <>
                  <Save size={16} /> Сохранить изменения
                </>
              )}
            </Button>
          </form>
        </section>

        <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <SectionHeader
            icon={Lock}
            title="Безопасность"
            eyebrow="Password"
          />
          <form onSubmit={submitChangePassword} className="grid gap-4 sm:grid-cols-3">
            <div>
              <label htmlFor="pwd-cur" className="input-label">Текущий пароль</label>
              <input
                id="pwd-cur"
                type="password"
                autoComplete="current-password"
                value={pwdCurrent}
                onChange={(e) => setPwdCurrent(e.target.value)}
                className="input-field"
              />
            </div>
            <div>
              <label htmlFor="pwd-new" className="input-label">Новый пароль</label>
              <input
                id="pwd-new"
                type="password"
                autoComplete="new-password"
                value={pwdNew}
                onChange={(e) => setPwdNew(e.target.value)}
                className="input-field"
              />
              {pwdNew && !pwdNewValid && (
                <p className="mt-1 text-xs text-amber-700">
                  Мин. 8 символов, латиница и цифры
                </p>
              )}
            </div>
            <div>
              <label htmlFor="pwd-conf" className="input-label">Подтверждение</label>
              <input
                id="pwd-conf"
                type="password"
                autoComplete="new-password"
                value={pwdConfirm}
                onChange={(e) => setPwdConfirm(e.target.value)}
                className="input-field"
              />
              {pwdConfirm && !pwdMatch && pwdNewValid && (
                <p className="mt-1 text-xs text-red-600">Пароли не совпадают</p>
              )}
            </div>
            {pwdMessage.text && (
              <p
                className={`sm:col-span-3 text-sm ${
                  pwdMessage.type === "ok" ? "text-emerald-700" : "text-red-600"
                }`}
              >
                {pwdMessage.text}
              </p>
            )}
            <div className="sm:col-span-3">
              <Button
                type="submit"
                disabled={pwdPending || !pwdCurrent || !pwdNewValid || !pwdMatch}
              >
                {pwdPending ? (
                  <>
                    <Loader2 size={16} className="animate-spin" /> Сохранение...
                  </>
                ) : (
                  <>
                    <ClipboardCheck size={16} /> Обновить пароль
                  </>
                )}
              </Button>
            </div>
          </form>
        </section>
      </div>

      <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <SectionHeader
            icon={Clock}
            title="История классификаций"
            eyebrow="Recent activity"
            className="mb-0"
          />
          {userInfo.emailVerified ? (
            <Button
              type="button"
              variant="secondary"
              onClick={fetchHistory}
              disabled={historyLoading}
              className="text-xs sm:ml-auto"
            >
              {historyLoading ? (
                <><Loader2 size={14} className="animate-spin" /> Загрузка...</>
              ) : (
                <><RefreshCw size={14} /> Обновить</>
              )}
            </Button>
          ) : (
            <p className="text-sm text-gray-400">
              Доступно после подтверждения email
            </p>
          )}
        </div>

        {history.length === 0 && !historyLoading && (
          <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 py-12 text-center">
            <FileText size={34} className="mx-auto mb-3 text-slate-300" />
            <p className="text-sm text-slate-500">
              {userInfo.emailVerified
                ? "Нажмите «Обновить», чтобы загрузить историю"
                : "Подтвердите email для доступа к истории"}
            </p>
          </div>
        )}

        {history.length > 0 && (
          <div className="space-y-3">
            {history.map((row, idx) => {
              const result = parseResult(row.result)
              const hasDetail = Object.prototype.hasOwnProperty.call(result, "detail")
              const hasResult = !hasDetail && Object.keys(result).length > 0
              const inProgress = row.status === "pending" || row.status === "processing"
              const isError = row.status === "error"
              const descriptionPending =
                row.description_status &&
                row.description_status !== "completed" &&
                row.description_status !== "error"
              const descriptionReady =
                row.description_status === "completed" && Boolean(row.description)
              const descriptionFailed =
                row.description_status === "error" || Boolean(row.description_error)
              const hasDescriptionInfo =
                row.description ||
                row.description_error ||
                (Array.isArray(row.important_labels) &&
                  row.important_labels.length > 0) ||
                (Array.isArray(row.bucketed_labels) &&
                  row.bucketed_labels.length > 0)
              const rowKey = `${row.request_date}_${row.file_name}_${idx}`
              const base = env.BACKEND_URL.replace(/\/$/, "")
              const imgSrc = row.image_token
                ? `${base}${HISTORY_IMAGE}?token=${encodeURIComponent(row.image_token)}`
                : null

              return (
                <div key={rowKey} className="rounded-lg border border-gray-200 bg-gray-50/50 p-4">
                  <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4">
                    <div className="flex items-center gap-2 text-xs text-gray-500 flex-shrink-0">
                      <Clock size={14} />
                      <span>{formatDate(row.request_date)}</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm font-medium text-gray-800 truncate">
                      <FileText size={14} className="text-gray-400 flex-shrink-0" />
                      <span className="truncate">{extractFileName(row.file_name)}</span>
                    </div>
                    <div className="sm:ml-auto flex items-center gap-2">
                      {inProgress && (
                        <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-medium text-amber-700">
                          <Loader2 size={12} className="animate-spin" /> В обработке
                        </span>
                      )}
                      {isError && (
                        <span className="inline-flex items-center gap-1 rounded-full bg-red-100 px-2.5 py-0.5 text-xs font-medium text-red-700">
                          <AlertTriangle size={12} /> Ошибка
                        </span>
                      )}
                      {!isError && !inProgress && hasResult && (
                        <span className="inline-flex items-center gap-1 rounded-full bg-green-100 px-2.5 py-0.5 text-xs font-medium text-green-700">
                          <CheckCircle2 size={12} /> Готово
                        </span>
                      )}
                      {descriptionPending && (
                        <span className="inline-flex items-center gap-1 rounded-full bg-blue-100 px-2.5 py-0.5 text-xs font-medium text-blue-700">
                          <Loader2 size={12} className="animate-spin" /> Описание готовится
                        </span>
                      )}
                      {descriptionReady && (
                        <span className="inline-flex items-center gap-1 rounded-full bg-med-50 px-2.5 py-0.5 text-xs font-medium text-med-700">
                          <FileText size={12} /> Описание готово
                        </span>
                      )}
                      {descriptionFailed && (
                        <span className="inline-flex items-center gap-1 rounded-full bg-red-100 px-2.5 py-0.5 text-xs font-medium text-red-700">
                          <AlertTriangle size={12} /> Ошибка описания
                        </span>
                      )}
                    </div>
                  </div>

                  {!isError && !inProgress && hasResult && (
                    <div className="mt-3">
                      <div className="flex flex-wrap gap-1.5">
                        {getValues(result).map((val, i) => (
                          <React.Fragment key={i}>
                            {i > 0 && <span className="self-center text-xs text-gray-300">&rarr;</span>}
                            <span className="rounded bg-med-50 px-2 py-0.5 text-xs font-medium text-med-800">
                              {val}
                            </span>
                          </React.Fragment>
                        ))}
                      </div>

                      <div className="mt-3 flex flex-wrap gap-2">
                        <button
                          type="button"
                          onClick={() => setOpenTreeKey((k) => (k === rowKey ? null : rowKey))}
                          className="inline-flex items-center gap-1.5 rounded-md bg-white border border-gray-200 px-3 py-1.5 text-xs font-medium text-gray-600 hover:bg-gray-50 transition-colors"
                        >
                          {openTreeKey === rowKey ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                          Дерево решений
                        </button>
                        {imgSrc && (
                          <button
                            type="button"
                            onClick={() => setOpenHistoryImageKey((k) => (k === rowKey ? null : rowKey))}
                            className="inline-flex items-center gap-1.5 rounded-md bg-white border border-gray-200 px-3 py-1.5 text-xs font-medium text-gray-600 hover:bg-gray-50 transition-colors"
                          >
                            <ImageIcon size={14} />
                            {openHistoryImageKey === rowKey ? "Скрыть" : "Изображение"}
                          </button>
                        )}
                        {hasDescriptionInfo && (
                          <button
                            type="button"
                            onClick={() => setOpenDescriptionKey((k) => (k === rowKey ? null : rowKey))}
                            className="inline-flex items-center gap-1.5 rounded-md bg-white border border-gray-200 px-3 py-1.5 text-xs font-medium text-gray-600 hover:bg-gray-50 transition-colors"
                          >
                            {openDescriptionKey === rowKey ? (
                              <ChevronUp size={14} />
                            ) : (
                              <ChevronDown size={14} />
                            )}
                            Описание
                          </button>
                        )}
                      </div>

                      {openTreeKey === rowKey && (
                        <div className="mt-3 animate-fadeIn">
                          <TreeComponent
                            classificationResult={result}
                            displaySize={{ width: "100%", height: "300px" }}
                            nodeSize={{ x: 300, y: 50 }}
                            zoom={0.4}
                            translate={{ x: 50, y: 180 }}
                          />
                        </div>
                      )}

                      {openHistoryImageKey === rowKey && imgSrc && (
                        <div className="mt-3 animate-fadeIn">
                          <img
                            src={imgSrc}
                            alt={row.file_name || "Снимок"}
                            className="max-h-80 w-auto max-w-full rounded-lg border border-gray-200 object-contain"
                            onError={() => setOpenHistoryImageKey(null)}
                          />
                        </div>
                      )}

                      {hasDescriptionInfo && openDescriptionKey === rowKey && (
                        <div className="mt-4 animate-fadeIn rounded-lg border border-gray-200 bg-white p-4">
                          <div className="flex items-center gap-2">
                            <FileText size={14} className="text-med-600" />
                            <p className="text-sm font-medium text-gray-800">
                              Клиническое описание
                            </p>
                          </div>

                          {row.description && (
                            <p className="mt-2 text-sm leading-relaxed text-gray-700 whitespace-pre-line">
                              {row.description}
                            </p>
                          )}

                          {Array.isArray(row.important_labels) &&
                            row.important_labels.length > 0 && (
                              <div className="mt-3 flex flex-wrap gap-2">
                                {row.important_labels.map((label) => (
                                  <span
                                    key={label}
                                    className="rounded-full bg-gray-100 px-2.5 py-1 text-xs font-medium text-gray-600"
                                  >
                                    {formatFeatureLabelText(label)}
                                  </span>
                                ))}
                              </div>
                            )}

                          <BucketLabelsDisclosure
                            labels={row.bucketed_labels}
                            className="mt-3"
                          />

                          {row.description_error && (
                            <p className="mt-2 text-sm text-red-600">
                              {row.description_error}
                            </p>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {!isError && !inProgress && hasDetail && (
                    <p className="mt-2 text-sm text-red-600">
                      Ошибка обработки. Обратитесь к администрации.
                    </p>
                  )}

                  {isError && (
                    <p className="mt-2 text-sm text-red-600">
                      Произошла ошибка. Обратитесь к администрации.
                    </p>
                  )}

                  {inProgress && (
                    <p className="mt-2 text-sm text-amber-700">
                      Классификация выполняется. Обновите историю позже.
                    </p>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </section>
    </div>
  )
}

export default Profile
