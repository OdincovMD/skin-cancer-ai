import React, { useState } from "react"
import { LogIn } from "lucide-react"

import { env } from "../../imports/ENV"
import {
  createVkPkceSession,
  formatVkSdkError,
  isVkIdConfigured,
} from "../../imports/vkId"

const VK_ID_SDK_URL = "https://unpkg.com/@vkid/sdk@2.6.1/dist-sdk/umd/index.js"

function getVkIdSdk() {
  return window.VKIDSDK || null
}

function loadVkIdSdk() {
  const readySdk = getVkIdSdk()
  if (readySdk) {
    return Promise.resolve(readySdk)
  }

  const existingScript = document.querySelector('script[data-vkid-sdk="true"]')
  if (existingScript) {
    return new Promise((resolve, reject) => {
      existingScript.addEventListener("load", () => resolve(getVkIdSdk()), {
        once: true,
      })
      existingScript.addEventListener(
        "error",
        () => reject(new Error("Не удалось загрузить VK ID SDK.")),
        { once: true }
      )
    })
  }

  return new Promise((resolve, reject) => {
    const script = document.createElement("script")
    script.src = VK_ID_SDK_URL
    script.async = true
    script.dataset.vkidSdk = "true"
    script.onload = () => {
      const sdk = getVkIdSdk()
      if (!sdk) {
        reject(new Error("VK ID SDK недоступен после загрузки."))
        return
      }
      resolve(sdk)
    }
    script.onerror = () => reject(new Error("Не удалось загрузить VK ID SDK."))
    document.head.appendChild(script)
  })
}

const VkIdButton = ({
  rememberMe = false,
  className = "",
  onError,
  children = "Войти через VK ID",
}) => {
  const [isPending, setIsPending] = useState(false)

  const handleClick = async () => {
    if (isPending) return

    if (!isVkIdConfigured()) {
      onError?.("VK ID ещё не настроен для этого окружения.")
      return
    }

    const appId = Number(env.VK_ID_APP_ID)
    if (!Number.isFinite(appId) || appId <= 0) {
      onError?.("Некорректная конфигурация VK ID.")
      return
    }

    try {
      setIsPending(true)
      const { state, codeVerifier } = createVkPkceSession(rememberMe)
      const VKID = await loadVkIdSdk()

      VKID.Config.init({
        app: appId,
        redirectUrl: env.VK_ID_REDIRECT_URI,
        source: VKID.ConfigSource.LOWCODE,
        state,
        codeVerifier,
        scope: env.VK_ID_SCOPES,
      })

      await VKID.Auth.login({
        authMode: VKID.ConfigAuthMode.Redirect,
      })
    } catch (err) {
      setIsPending(false)
      onError?.(`Ошибка VK ID: ${formatVkSdkError(err)}`)
    }
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={isPending}
      className={`inline-flex w-full items-center justify-center gap-2 rounded-xl border border-[#0077FF]/18 bg-[#0077FF] px-4 py-3 text-sm font-semibold text-white shadow-[0_18px_32px_-24px_rgba(0,119,255,0.9)] transition hover:-translate-y-0.5 hover:bg-[#0069e0] hover:shadow-[0_22px_38px_-24px_rgba(0,119,255,0.95)] disabled:translate-y-0 disabled:cursor-not-allowed disabled:opacity-60 ${className}`.trim()}
    >
      <LogIn size={18} />
      {isPending ? "Переходим в VK ID..." : children}
    </button>
  )
}

export default VkIdButton
