import React, { useEffect, useRef, useState } from "react"
import { useDispatch, useSelector } from "react-redux"
import { handleHistoryRequest } from "../asyncActions/handleHistoryRequest"

import { env } from "../imports/ENV"
import { bearerAuthHeaders } from "../imports/authHeaders"
import {
  CHANGE_PASSWORD,
  HISTORY_IMAGE,
  ME_AVATAR,
  ME_PROFILE,
  RESEND_VERIFICATION_EMAIL,
} from "../imports/ENDPOINTS"
import { getValues, mappingInfoRU } from "../imports/HELPERS"

import TreeComponent from "../components/Tree"
import {
  mergeUserData,
  setVerificationResendCooldownFromSeconds,
} from "../store/userReducer"

const Profile = () => {
  const dispatch = useDispatch()
  const userInfo = useSelector(state => state.user)

  // request_date, file_name, bucket_name?, status, result
  const [history, setHistory] = useState([])
  const [openHistoryImageKey, setOpenHistoryImageKey] = useState(null)
  const [resendPending, setResendPending] = useState(false)
  const [resendHint, setResendHint] = useState("")
  const [, tickResendCooldown] = useState(0)
  const verificationResendUntilMs = userInfo.verificationResendUntilMs

  useEffect(() => {
    if (!verificationResendUntilMs || Date.now() >= verificationResendUntilMs) {
      return undefined
    }
    const id = window.setInterval(() => {
      tickResendCooldown((n) => n + 1)
    }, 1000)
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

  const avatarObjectUrlRef = useRef(null)
  const [avatarDisplayUrl, setAvatarDisplayUrl] = useState(null)
  const [avatarVersion, setAvatarVersion] = useState(0)
  const [editFirstName, setEditFirstName] = useState("")
  const [editLastName, setEditLastName] = useState("")
  const [profilePending, setProfilePending] = useState(false)
  const [profileMessage, setProfileMessage] = useState({ type: "", text: "" })
  const [avatarUploadPending, setAvatarUploadPending] = useState(false)
  const [avatarMessage, setAvatarMessage] = useState({ type: "", text: "" })

  useEffect(() => {
    setEditFirstName(userInfo.userData?.firstName ?? "")
    setEditLastName(userInfo.userData?.lastName ?? "")
  }, [
    userInfo.userData?.id,
    userInfo.userData?.firstName,
    userInfo.userData?.lastName,
  ])

  useEffect(() => {
    if (!userInfo.accessToken) {
      if (avatarObjectUrlRef.current) {
        URL.revokeObjectURL(avatarObjectUrlRef.current)
        avatarObjectUrlRef.current = null
      }
      setAvatarDisplayUrl(null)
      return undefined
    }
    const ctrl = new AbortController()
    const base = env.BACKEND_URL.replace(/\/$/, "")
    ;(async () => {
      try {
        const res = await fetch(`${base}${ME_AVATAR}`, {
          headers: {
            accept: "image/*",
            ...bearerAuthHeaders(userInfo.accessToken),
          },
          signal: ctrl.signal,
        })
        if (ctrl.signal.aborted) return
        if (res.status === 404) {
          if (avatarObjectUrlRef.current) {
            URL.revokeObjectURL(avatarObjectUrlRef.current)
            avatarObjectUrlRef.current = null
          }
          setAvatarDisplayUrl(null)
          return
        }
        if (!res.ok) return
        const blob = await res.blob()
        if (ctrl.signal.aborted) return
        if (avatarObjectUrlRef.current) {
          URL.revokeObjectURL(avatarObjectUrlRef.current)
        }
        const url = URL.createObjectURL(blob)
        avatarObjectUrlRef.current = url
        setAvatarDisplayUrl(url)
      } catch (e) {
        if (e.name !== "AbortError") {
          /* сеть / прочее */
        }
      }
    })()
    return () => {
      ctrl.abort()
      if (avatarObjectUrlRef.current) {
        URL.revokeObjectURL(avatarObjectUrlRef.current)
        avatarObjectUrlRef.current = null
      }
      setAvatarDisplayUrl(null)
    }
  }, [userInfo.accessToken, avatarVersion])

  // useEffect(() => {
  //   console.log(history)
  // }, [history])

  const showInfo = (field) => {
    return (
      mappingInfoRU[field] ?
        <div className="flex flex-row justify-start items-center gap-[10px]">
          <div className="rounded-lg border-none w-[170px] p-3">
            <span className="block truncate font-semibold">{mappingInfoRU[field]}</span>
          </div>
          <div className="rounded-lg border flex-grow border-gray-300 p-3">
            <span className="block truncate">{userInfo.userData[field]}</span>
          </div>
        </div> :
        null
    )
  }
  
  const showHistory = (historyResponse, userId) => {
    const isHeaderRow = historyResponse.file_name === mappingInfoRU.file_name
    const rowKey = `${String(historyResponse.request_date)}_${historyResponse.file_name}`
    const base = env.BACKEND_URL.replace(/\/$/, "")
    const imageSrc =
      userId &&
      historyResponse.file_name &&
      !isHeaderRow &&
      historyResponse.image_token
        ? `${base}${HISTORY_IMAGE}?token=${encodeURIComponent(
            historyResponse.image_token
          )}`
        : null

    const data_time = new RegExp("^(?<data>.*)T(?<time>.*)\\..*\\+(?<correction>.*)$")
    const file_name = new RegExp("^(?:.*?_){3}(?<filename>.*)$")

    const requestDate = data_time.exec(historyResponse.request_date)
    const fileName = file_name.exec(historyResponse.file_name)

    const parseClassificationResult = () => {
      const raw = historyResponse.result
      if (raw == null || raw === "") return {}
      try {
        const parsed = typeof raw === "string" ? JSON.parse(raw) : raw
        if (parsed != null && typeof parsed === "object" && !Array.isArray(parsed)) {
          return parsed
        }
        return {}
      } catch {
        return {}
      }
    }

    const result = parseClassificationResult()
    const status = historyResponse.status
    const inProgress =
      status === "pending" || status === "processing"

    return (
      <div className="rounded-lg border border-gray-900 p-3 space-y-3">
      <ul className="flex flex-row justify-between items-center">
        <li 
          key={0}
          className="w-[15%] text-center"
        >
          {
            requestDate ? 
            `${requestDate.groups.data}, ${requestDate.groups.time}` :
            mappingInfoRU.request_date
          }
        </li>
        <li 
          key={1}
          className="w-[10%] text-center"
        >
          {
            fileName ? 
            `${fileName.groups.filename}` :
            mappingInfoRU.file_name
          }
        </li>
        <li 
          key={3}
          className="flex flex-col justify-center items-center gap-[10px] w-[55%] text-center"
        > 
          {
          (status === "error") &&
          <p className="text-red-600">
            Произошла ошибка со стороны бэкенда. Свяжитесь с администрацией сайта.
          </p>
          }
          {
          inProgress &&
          <p className="text-amber-700">
            Классификация выполняется или ожидает обработки. Обновите историю позже.
          </p>
          }
          {
          (status !== "error" && !inProgress) && Object.prototype.hasOwnProperty.call(result, "detail") &&
          <p className="text-red-600">
            Произошла ошибка обработки фотографии. Свяжитесь с администрацией сайта.
          </p>
          }
          {
          (status !== "error" && !inProgress) && !Object.prototype.hasOwnProperty.call(result, "detail") && (Object.keys(result).length > 0) &&
          <div className="w-[100%] font-semibold text-gray-700">
            {getValues(result).reduce((accumulator, currentValue) => (accumulator + " ->\n" + currentValue))}
          </div>
          }
          {
          (status !== "error" && !inProgress) &&
          <div className="flex flex-col justify-center items-center w-[100%]">
            {
              !Object.prototype.hasOwnProperty.call(result, "detail") ? (
                Object.keys(result).length > 0 ? (
                  <TreeComponent classificationResult={result} displaySize={{width: "100%", height: "300px"}} nodeSize={{x: 300, y: 50}} zoom={0.4} translate={{x: 50, y: 180}}/>
                ) : (
                  mappingInfoRU.result
                )
              ) : null
            }
          </div>
          }
        </li>
      </ul>
      {!isHeaderRow && imageSrc && (
        <div className="flex flex-col items-center gap-2 border-t border-gray-200 pt-3">
          <button
            type="button"
            onClick={() =>
              setOpenHistoryImageKey((k) => (k === rowKey ? null : rowKey))
            }
            className="px-4 py-2 text-sm bg-slate-600 text-white rounded-md hover:bg-slate-700"
          >
            {openHistoryImageKey === rowKey
              ? "Скрыть изображение"
              : "Показать изображение из хранилища"}
          </button>
          {openHistoryImageKey === rowKey && (
            <img
              src={imageSrc}
              alt={historyResponse.file_name || "Снимок"}
              className="max-h-[min(480px,70vh)] w-auto max-w-full rounded border border-gray-300 object-contain"
              onError={() => {
                alert("Не удалось загрузить изображение (файл отсутствует в MinIO или ошибка сети).")
                setOpenHistoryImageKey(null)
              }}
            />
          )}
        </div>
      )}
      </div>
    )
  }

  const defaultProfileImg =
    userInfo.userData?.id == 1 ? "/images/PP.png" : "/images/image.png"

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
        headers: {
          ...bearerAuthHeaders(userInfo.accessToken),
        },
        body: fd,
      })
      const data = await res.json().catch(() => ({}))
      if (res.status === 401) {
        setAvatarMessage({
          type: "err",
          text: "Сессия истекла. Войдите снова.",
        })
        return
      }
      if (data.error) {
        setAvatarMessage({ type: "err", text: data.error })
        return
      }
      setAvatarMessage({
        type: "ok",
        text: "Фото профиля обновлено.",
      })
      setAvatarVersion((v) => v + 1)
    } catch (err) {
      setAvatarMessage({
        type: "err",
        text: String(err?.message || err),
      })
    } finally {
      setAvatarUploadPending(false)
    }
  }

  const profilePicture =
    userInfo.accessToken && userInfo.userData?.id ? (
      <div className="flex w-[220px] flex-shrink-0 flex-col space-y-3">
        <div className="h-[220px] rounded-lg bg-white p-4 shadow-md">
          <img
            src={avatarDisplayUrl || defaultProfileImg}
            alt="Фотография профиля"
            className="h-full w-full rounded-lg border border-gray-700 object-cover"
          />
        </div>
        <label className="block cursor-pointer rounded-lg border border-gray-300 bg-white px-3 py-2 text-center text-sm font-medium text-gray-800 shadow-sm transition hover:bg-gray-50">
          <input
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
            disabled={avatarUploadPending}
            onChange={handleAvatarFile}
          />
          {avatarUploadPending ? "Загрузка…" : "Загрузить фото"}
        </label>
        {avatarMessage.text && (
          <p
            className={
              avatarMessage.type === "ok"
                ? "text-center text-xs text-green-800"
                : "text-center text-xs text-red-600"
            }
          >
            {avatarMessage.text}
          </p>
        )}
      </div>
    ) : (
      <div className="flex-shrink-0 space-y-6">
        <div className="h-[220px] rounded-lg bg-white p-6 shadow-md">
          <img
            src={defaultProfileImg}
            alt="Фотография профиля"
            className="h-full w-full rounded-lg border border-gray-700 object-cover"
          />
        </div>
      </div>
    )

  const profileFields = Object.keys(userInfo.userData || {}).filter(
    (field) => mappingInfoRU[field]
  )

  const profileInfo =
    <div className="space-y-6">
      <div className="flex flex-row items-center gap-[10px] bg-white rounded-lg shadow-md p-6">
        <ul className="space-y-2 flex-grow">
          {profileFields.map((field, index) => (
            <li key={index} >
              {showInfo(field)}
            </li>
          ))}
        </ul>
      </div>
    </div>

  const submitProfileNames = async (e) => {
    e.preventDefault()
    setProfileMessage({ type: "", text: "" })
    const fn = editFirstName.trim()
    const ln = editLastName.trim()
    const body = {}
    if (fn) body.firstName = fn
    if (ln) body.lastName = ln
    if (Object.keys(body).length === 0) {
      setProfileMessage({
        type: "err",
        text: "Укажите имя или фамилию.",
      })
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
        setProfileMessage({
          type: "err",
          text: "Сессия истекла. Войдите снова.",
        })
        return
      }
      if (res.status === 422) {
        const d = data.detail
        const msg =
          Array.isArray(d) && d[0]?.msg != null
            ? String(d[0].msg)
            : "Проверьте введённые данные."
        setProfileMessage({ type: "err", text: msg })
        return
      }
      if (data.error) {
        setProfileMessage({ type: "err", text: data.error })
        return
      }
      if (data.userData) {
        dispatch(mergeUserData(data.userData))
      }
      setProfileMessage({ type: "ok", text: "Данные сохранены." })
    } catch (err) {
      setProfileMessage({
        type: "err",
        text: String(err?.message || err),
      })
    } finally {
      setProfilePending(false)
    }
  }

  const profileNamesCard =
    userInfo.accessToken && userInfo.userData?.id ? (
      <div className="mt-6 w-full max-w-2xl rounded-lg border border-gray-200 bg-white p-6 shadow-md">
        <h2 className="mb-4 text-lg font-semibold text-gray-900">
          Имя и фамилия
        </h2>
        <form
          onSubmit={submitProfileNames}
          className="flex max-w-md flex-col gap-3"
        >
          <label className="flex flex-col gap-1 text-sm">
            <span className="font-medium text-gray-700">Имя</span>
            <input
              type="text"
              autoComplete="given-name"
              value={editFirstName}
              onChange={(ev) => setEditFirstName(ev.target.value)}
              maxLength={100}
              className="rounded-md border border-gray-300 px-3 py-2"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="font-medium text-gray-700">Фамилия</span>
            <input
              type="text"
              autoComplete="family-name"
              value={editLastName}
              onChange={(ev) => setEditLastName(ev.target.value)}
              maxLength={100}
              className="rounded-md border border-gray-300 px-3 py-2"
            />
          </label>
          {profileMessage.text && (
            <p
              className={
                profileMessage.type === "ok"
                  ? "text-sm text-green-800"
                  : "text-sm text-red-600"
              }
            >
              {profileMessage.text}
            </p>
          )}
          <button
            type="submit"
            disabled={profilePending}
            className="mt-1 w-fit rounded-lg bg-slate-700 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:bg-gray-400"
          >
            {profilePending ? "Сохранение…" : "Сохранить"}
          </button>
        </form>
      </div>
    ) : null

  const submitChangePassword = async (e) => {
    e.preventDefault()
    setPwdMessage({ type: "", text: "" })
    if (!pwdNewValid) {
      setPwdMessage({
        type: "err",
        text: "Новый пароль: не менее 8 символов, только латиница и цифры.",
      })
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
        body: JSON.stringify({
          current_password: pwdCurrent,
          new_password: pwdNew,
        }),
      })
      const data = await res.json().catch(() => ({}))
      if (res.status === 401) {
        setPwdMessage({
          type: "err",
          text: "Сессия истекла. Войдите снова.",
        })
        return
      }
      if (res.status === 422) {
        const d = data.detail
        const msg =
          Array.isArray(d) && d[0]?.msg != null
            ? String(d[0].msg)
            : "Проверьте введённые данные."
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
      setPwdMessage({ type: "ok", text: "Пароль успешно изменён." })
    } catch (err) {
      setPwdMessage({
        type: "err",
        text: String(err?.message || err),
      })
    } finally {
      setPwdPending(false)
    }
  }

  const changePasswordCard =
    userInfo.accessToken && userInfo.userData?.id ? (
      <div className="mt-6 w-full max-w-2xl rounded-lg border border-gray-200 bg-white p-6 shadow-md">
        <h2 className="mb-4 text-lg font-semibold text-gray-900">
          Смена пароля
        </h2>
        <form onSubmit={submitChangePassword} className="flex max-w-md flex-col gap-3">
          <label className="flex flex-col gap-1 text-sm">
            <span className="font-medium text-gray-700">Текущий пароль</span>
            <input
              type="password"
              autoComplete="current-password"
              value={pwdCurrent}
              onChange={(ev) => setPwdCurrent(ev.target.value)}
              className="rounded-md border border-gray-300 px-3 py-2"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="font-medium text-gray-700">Новый пароль</span>
            <input
              type="password"
              autoComplete="new-password"
              value={pwdNew}
              onChange={(ev) => setPwdNew(ev.target.value)}
              className="rounded-md border border-gray-300 px-3 py-2"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="font-medium text-gray-700">
              Повторите новый пароль
            </span>
            <input
              type="password"
              autoComplete="new-password"
              value={pwdConfirm}
              onChange={(ev) => setPwdConfirm(ev.target.value)}
              className="rounded-md border border-gray-300 px-3 py-2"
            />
          </label>
          {pwdNew && !pwdNewValid && (
            <p className="text-sm text-amber-800">
              Не менее 8 символов, только латиница и цифры.
            </p>
          )}
          {pwdConfirm && !pwdMatch && pwdNewValid && (
            <p className="text-sm text-red-600">Пароли не совпадают.</p>
          )}
          {pwdMessage.text && (
            <p
              className={
                pwdMessage.type === "ok"
                  ? "text-sm text-green-800"
                  : "text-sm text-red-600"
              }
            >
              {pwdMessage.text}
            </p>
          )}
          <button
            type="submit"
            disabled={
              pwdPending ||
              !pwdCurrent ||
              !pwdNewValid ||
              !pwdMatch
            }
            className="mt-1 w-fit rounded-lg bg-slate-700 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:bg-gray-400"
          >
            {pwdPending ? "Сохранение…" : "Сохранить новый пароль"}
          </button>
        </form>
      </div>
    ) : null

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
        if (data.retry_after_seconds != null) {
          dispatch(
            setVerificationResendCooldownFromSeconds(data.retry_after_seconds)
          )
        }
        setResendHint(data.error)
      } else {
        setResendHint("Письмо отправлено. Проверьте почту и папку «Спам».")
        dispatch(
          setVerificationResendCooldownFromSeconds(
            data.verification_resend_after_seconds ?? 120
          )
        )
      }
    } catch (e) {
      alert(String(e.message || e))
    } finally {
      setResendPending(false)
    }
  }

  const emailBanner =
    userInfo.accessToken && !userInfo.emailVerified ? (
      <div className="mb-6 w-full max-w-2xl rounded-lg border border-amber-300 bg-amber-50 p-5 text-amber-950 shadow-sm">
        <p className="mb-3 font-semibold">Адрес email не подтверждён</p>
        <p className="mb-4 text-sm">
          Загрузка изображений в классификатор и просмотр истории классификаций будут доступны после
          перехода по ссылке из письма. Если письма нет, отправьте его повторно.
        </p>
        <button
          type="button"
          onClick={resendVerification}
          disabled={resendPending || resendCooldownRemaining > 0}
          className="rounded-lg bg-amber-800 px-4 py-2 text-sm font-semibold text-white transition hover:bg-amber-900 disabled:bg-gray-400"
        >
          {resendPending
            ? "Отправка…"
            : resendCooldownRemaining > 0
              ? `Повторная отправка через ${resendCooldownRemaining} с`
              : "Отправить письмо повторно"}
        </button>
        {resendHint && (
          <p
            className={`mt-3 text-sm ${
              resendHint.startsWith("Письмо отправлено")
                ? "text-green-800"
                : "text-amber-900"
            }`}
          >
            {resendHint}
          </p>
        )}
      </div>
    ) : null

  const historyDisplay = history.length > 0 ?
    <div className="space-y-6 mt-5">
      <div className="flex flex-column items-center justify-center bg-white rounded-lg shadow-md p-6">
        <ul className="w-full space-y-2">
          {history.map((historyResponse, index) => (
            <li key={`${String(historyResponse.request_date)}_${historyResponse.file_name}_${index}`}>
              {showHistory(historyResponse, userInfo.userData.id)}
            </li>
          ))}
        </ul>
      </div>
    </div> :
    null

  const requestButton =
    userInfo.emailVerified ? (
    <div className="space-y-6 mt-5 w-full">
      <div className="flex flex-column items-center justify-center bg-white rounded-lg shadow-md p-6">
        <div>
            <button 
              onClick={() => {
                handleHistoryRequest(userInfo.accessToken).then((response) => {
                  const rows = Array.isArray(response) ? response : []
                  setOpenHistoryImageKey(null)
                  setHistory([
                    {
                      request_date: mappingInfoRU.request_date,
                      file_name: mappingInfoRU.file_name,
                      status: mappingInfoRU.status,
                      result: mappingInfoRU.result,
                    },
                    ...rows,
                  ])
                })
              }} 
              className="px-6 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
            >
              Получить историю запросов
            </button>
        </div>
      </div>
    </div>
    ) : (
    <div className="mt-5 w-full max-w-xl rounded-lg border border-gray-200 bg-white p-6 text-center text-gray-600 shadow-md">
      История классификаций станет доступна после подтверждения email.
    </div>
    )

  return (
    <div className="flex flex-col justify-center items-center">
      {emailBanner}
      <div className="flex flex-row justify-center gap-[20px] w-[60%]">
          {profilePicture}
          {profileInfo}
      </div>
      <div className="flex flex-col justify-center items-center w-[80%]">
        {profileNamesCard}
        {changePasswordCard}
        {historyDisplay}
        {requestButton}
      </div>
    </div>
  )
}

export default Profile