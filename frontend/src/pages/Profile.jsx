import React, { useCallback, useEffect, useState } from "react"
import { useDispatch, useSelector } from "react-redux"
import { Link } from "react-router-dom"
import {
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
  Loader2,
  User,
  ImageIcon,
  RefreshCw,
  Key,
  Copy,
  Trash2,
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

const SectionHeader = ({ icon: Icon, title }) => (
  <div className="flex items-center gap-2 mb-4">
    <Icon size={18} className="text-med-600" />
    <h2 className="text-base font-semibold text-gray-900">{title}</h2>
  </div>
)

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

  const extractFileName = (raw) => {
    const m = /^(?:.*?_){3}(?<filename>.*)$/.exec(raw)
    return m?.groups?.filename ?? raw
  }

  return (
    <div className="space-y-6">
      {userInfo.accessToken && !userInfo.emailVerified && (
        <div className="flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 px-5 py-4">
          <Mail size={20} className="mt-0.5 flex-shrink-0 text-amber-600" />
          <div className="flex-1">
            <p className="font-semibold text-amber-900 text-sm">
              Адрес email не подтверждён
            </p>
            <p className="mt-1 text-sm text-amber-800">
              Загрузка и история будут доступны после перехода по ссылке из письма.
            </p>
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <Button
                type="button"
                onClick={resendVerification}
                disabled={resendPending || resendCooldownRemaining > 0}
                className="text-xs bg-amber-700 hover:bg-amber-800 focus-visible:ring-amber-500"
              >
                {resendPending
                  ? "Отправка..."
                  : resendCooldownRemaining > 0
                    ? `Повтор через ${resendCooldownRemaining} с`
                    : "Отправить письмо"}
              </Button>
              {resendHint && (
                <p className={`text-xs ${resendHint.startsWith("Письмо") ? "text-green-700" : "text-amber-900"}`}>
                  {resendHint}
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="card-elevated">
        <div className="flex flex-col sm:flex-row items-center sm:items-start gap-6">
          {/* Avatar */}
          <div className="relative group">
            <div className="h-24 w-24 rounded-full overflow-hidden bg-med-100 flex items-center justify-center ring-4 ring-white shadow">
              {avatarDisplayUrl ? (
                <img
                  src={avatarDisplayUrl}
                  alt="Аватар"
                  className="h-full w-full object-cover"
                />
              ) : (
                <span className="text-2xl font-bold text-med-600">
                  {initials || <User size={32} />}
                </span>
              )}
            </div>
            <label className="absolute inset-0 flex cursor-pointer items-center justify-center rounded-full bg-black/40 opacity-0 transition-opacity group-hover:opacity-100">
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

          {/* Info */}
          <div className="flex-1 text-center sm:text-left">
            <h1 className="text-xl font-bold text-gray-900">
              {userInfo.userData?.firstName} {userInfo.userData?.lastName}
            </h1>
            <p className="mt-0.5 text-sm text-gray-500">
              {userInfo.userData?.email}
            </p>
            {userInfo.emailVerified && (
              <span className="mt-2 inline-flex items-center gap-1 rounded-full bg-green-50 px-2.5 py-0.5 text-xs font-medium text-green-700">
                <CheckCircle2 size={12} /> Email подтверждён
              </span>
            )}
            {avatarMessage.text && (
              <p className={`mt-2 text-xs ${avatarMessage.type === "ok" ? "text-green-700" : "text-red-600"}`}>
                {avatarMessage.text}
              </p>
            )}
          </div>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Edit name */}
        <div className="card">
          <SectionHeader icon={User} title="Личные данные" />
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
              <p className={`text-sm ${profileMessage.type === "ok" ? "text-green-700" : "text-red-600"}`}>
                {profileMessage.text}
              </p>
            )}
            <Button type="submit" disabled={profilePending}>
              {profilePending ? (
                <><Loader2 size={16} className="animate-spin" /> Сохранение...</>
              ) : (
                <><Save size={16} /> Сохранить</>
              )}
            </Button>
          </form>
        </div>

        {/* Change password */}
        <div className="card">
          <SectionHeader icon={Lock} title="Смена пароля" />
          <form onSubmit={submitChangePassword} className="space-y-4">
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
              <p className={`text-sm ${pwdMessage.type === "ok" ? "text-green-700" : "text-red-600"}`}>
                {pwdMessage.text}
              </p>
            )}
            <Button
              type="submit"
              disabled={pwdPending || !pwdCurrent || !pwdNewValid || !pwdMatch}
            >
              {pwdPending ? (
                <><Loader2 size={16} className="animate-spin" /> Сохранение...</>
              ) : (
                <><Lock size={16} /> Сменить пароль</>
              )}
            </Button>
          </form>
        </div>
      </div>

      {userInfo.emailVerified && (
        <div className="card-elevated">
          <SectionHeader icon={Key} title="Доступ по API" />
          <p className="text-sm text-gray-600 mb-4">
            Ключ для запросов к{" "}
            <Link to={API_DOCS} className="text-link">
              HTTP API v1
            </Link>{" "}
            (заголовок <code className="text-xs bg-gray-100 px-1 rounded">X-API-Key</code>
            ). Лимит на стороне сервера — 5 запросов в минуту.
          </p>
          {apiTokLoading && !apiTokStatus ? (
            <p className="text-sm text-gray-500 flex items-center gap-2">
              <Loader2 size={16} className="animate-spin" /> Загрузка…
            </p>
          ) : (
            <>
              {apiTokStatus?.has_token && (
                <p className="text-sm text-gray-700 mb-2">
                  <span className="font-medium">Статус:</span>{" "}
                  {apiTokStatus.display_label || "scai_••••••••"}
                  {apiTokStatus.created_at && (
                    <span className="text-gray-500 ml-2">
                      (создан {formatDate(apiTokStatus.created_at)})
                    </span>
                  )}
                </p>
              )}
              {apiTokPlain && (
                <div className="space-y-2 mb-4">
                  <label className="input-label">Ключ (показан один раз)</label>
                  <textarea
                    readOnly
                    className="input-field font-mono text-xs min-h-[4.5rem]"
                    value={apiTokPlain}
                  />
                  <Button
                    type="button"
                    variant="secondary"
                    className="text-xs"
                    onClick={copyApiToken}
                  >
                    <Copy size={14} /> Копировать
                  </Button>
                </div>
              )}
              <div className="flex flex-wrap gap-2">
                {!apiTokStatus?.has_token ? (
                  <Button
                    type="button"
                    disabled={apiTokActionPending}
                    onClick={issueApiToken}
                  >
                    {apiTokActionPending ? (
                      <><Loader2 size={16} className="animate-spin" /> Выпуск…</>
                    ) : (
                      <><Key size={16} /> Выпустить ключ</>
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
                        <><Loader2 size={16} className="animate-spin" /> …</>
                      ) : (
                        <><RefreshCw size={16} /> Перевыпустить</>
                      )}
                    </Button>
                    <Button
                      type="button"
                      variant="secondary"
                      className="text-red-700 border-red-200 hover:bg-red-50"
                      disabled={apiTokActionPending}
                      onClick={revokeApiToken}
                    >
                      <Trash2 size={16} /> Отозвать
                    </Button>
                  </>
                )}
              </div>
              {apiTokMsg.text && (
                <p
                  className={`mt-3 text-sm ${
                    apiTokMsg.type === "ok" ? "text-green-700" : "text-red-600"
                  }`}
                >
                  {apiTokMsg.text}
                </p>
              )}
            </>
          )}
        </div>
      )}

      <div className="card-elevated">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
          <SectionHeader icon={Clock} title="История классификаций" />
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
          <div className="py-10 text-center">
            <FileText size={32} className="mx-auto mb-2 text-gray-300" />
            <p className="text-sm text-gray-400">
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
      </div>
    </div>
  )
}

export default Profile
