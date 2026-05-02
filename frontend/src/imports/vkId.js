import { env } from "./ENV"

const VK_CODE_VERIFIER_PREFIX = "vkid_code_verifier_"
const VK_LINK_TOKEN_KEY = "vkid_link_token"
const VK_LINK_EMAIL_KEY = "vkid_link_email"

function base64Url(bytes) {
  return btoa(String.fromCharCode(...bytes))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "")
}

function randomString(bytes = 32) {
  const raw = new Uint8Array(bytes)
  crypto.getRandomValues(raw)
  return base64Url(raw)
}

export function isVkIdConfigured() {
  return Boolean(env.VK_ID_APP_ID && env.VK_ID_REDIRECT_URI)
}

export function createVkPkceSession(rememberMe = false) {
  const state = randomString(24)
  const codeVerifier = randomString(48)
  sessionStorage.setItem(
    `${VK_CODE_VERIFIER_PREFIX}${state}`,
    JSON.stringify({ codeVerifier, rememberMe: Boolean(rememberMe) })
  )
  return { state, codeVerifier }
}

export function consumeVkCodeVerifier(state) {
  if (!state) return null
  const key = `${VK_CODE_VERIFIER_PREFIX}${state}`
  const value = sessionStorage.getItem(key)
  sessionStorage.removeItem(key)
  if (!value) return null
  try {
    const parsed = JSON.parse(value)
    if (typeof parsed?.codeVerifier === "string") {
      return {
        codeVerifier: parsed.codeVerifier,
        rememberMe: parsed.rememberMe === true,
      }
    }
  } catch {
  }
  return { codeVerifier: value, rememberMe: false }
}

export function stashVkLinkSession(token, email) {
  if (typeof token === "string" && token) {
    sessionStorage.setItem(VK_LINK_TOKEN_KEY, token)
  }
  if (typeof email === "string" && email) {
    sessionStorage.setItem(VK_LINK_EMAIL_KEY, email)
  }
}

export function readVkLinkSession() {
  return {
    vkLinkToken: sessionStorage.getItem(VK_LINK_TOKEN_KEY),
    email: sessionStorage.getItem(VK_LINK_EMAIL_KEY),
  }
}

export function clearVkLinkSession() {
  sessionStorage.removeItem(VK_LINK_TOKEN_KEY)
  sessionStorage.removeItem(VK_LINK_EMAIL_KEY)
}

function readParams(raw = "") {
  const normalized = raw.startsWith("?") || raw.startsWith("#") ? raw.slice(1) : raw
  return new URLSearchParams(normalized)
}

function decodePayload(payloadRaw) {
  if (!payloadRaw) {
    return {}
  }
  try {
    return JSON.parse(payloadRaw)
  } catch {
    return {}
  }
}

function firstNonEmpty(...values) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value
    }
  }
  return null
}

export function formatVkSdkError(error) {
  if (!error) {
    return "Неизвестная ошибка VK ID."
  }
  if (typeof error === "string") {
    return error
  }
  if (error instanceof Error && error.message) {
    return error.message
  }

  const code = firstNonEmpty(
    error.code,
    error.error,
    error.type,
    error.name
  )
  const message = firstNonEmpty(
    error.description,
    error.error_description,
    error.message,
    error.reason
  )

  if (code && message) {
    return `${code}: ${message}`
  }
  if (message) {
    return message
  }
  if (code) {
    return code
  }

  try {
    return JSON.stringify(error)
  } catch {
    return "Неизвестная ошибка VK ID."
  }
}

export function isVkSdkTimeoutError(error) {
  if (!error || typeof error !== "object") {
    return false
  }

  const code = Number(error.code)
  const text = typeof error.text === "string" ? error.text.trim().toLowerCase() : ""
  const message = typeof error.message === "string" ? error.message.trim().toLowerCase() : ""

  return code === 0 && (text === "timeout" || message === "timeout")
}

export function readVkCallbackParams(search, hash = "") {
  const searchParams = readParams(search)
  const hashParams = readParams(hash)
  const payload = {
    ...decodePayload(searchParams.get("payload")),
    ...decodePayload(hashParams.get("payload")),
  }

  return {
    code: firstNonEmpty(
      searchParams.get("code"),
      searchParams.get("auth_code"),
      hashParams.get("code"),
      hashParams.get("auth_code"),
      payload.code,
      payload.auth_code
    ),
    deviceId: firstNonEmpty(
      searchParams.get("device_id"),
      hashParams.get("device_id"),
      payload.device_id,
      payload.deviceId
    ),
    state: firstNonEmpty(
      searchParams.get("state"),
      hashParams.get("state"),
      payload.state
    ),
    error: firstNonEmpty(
      searchParams.get("error"),
      hashParams.get("error"),
      payload.error
    ),
    errorDescription: firstNonEmpty(
      searchParams.get("error_description"),
      hashParams.get("error_description"),
      payload.error_description,
      payload.errorDescription,
      payload.description
    ),
  }
}
