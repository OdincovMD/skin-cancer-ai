import { env } from "../imports/ENV"
import { createAsyncThunk } from "@reduxjs/toolkit"

async function messageFromFailedResponse(response) {
  try {
    const body = await response.json()
    if (body?.detail != null || body?.error != null) {
      let error = null
      const d = body?.detail
      if (d != null) {
        if (typeof d === "string") error = d
        else if (Array.isArray(d) && d[0]?.msg != null) error = String(d[0].msg)
        else error = JSON.stringify(d)
      } else if (body?.error != null && typeof body.error === "string") {
        error = body.error
      }
      return {
        error,
        requiresVkLink: body?.requires_vk_link === true,
        vkLinkToken:
          typeof body?.vk_link_token === "string" ? body.vk_link_token : null,
        email: typeof body?.email === "string" ? body.email : null,
      }
    }
  } catch {
  }
  const s = response.status
  let error = `Ошибка запроса (${s}). Повторите попытку.`
  if (s === 401) error = "Неверный email или пароль."
  if (s === 403) error = "Доступ запрещён."
  if (s === 422) error = "Проверьте введённые данные."
  if (s === 503) error = "Сервис временно недоступен. Попробуйте позже."
  if (s >= 500) error = "Ошибка сервера. Попробуйте позже."
  return {
    error,
    requiresVkLink: false,
    vkLinkToken: null,
    email: null,
  }
}

export const onVerify = createAsyncThunk("user/onVerify", async ({data, endpoint}) => {

    let requestState = {
      userData: {
        id: null,
        firstName: null,
        lastName: null,
        email: null,
        hasPassword: false,
      },
      accessToken: null,
      accessTokenExpiresAt: null,
      error: null,
      requires_email_verification: false,
      emailVerified: true,
      verificationResendAfterSeconds: 0,
      requiresVkLink: false,
      vkLinkToken: null,
      email: null,
    }

    try {
      let response = await fetch(`${env.BACKEND_URL}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-type": "application/json",
          "accept": "application/json"
        },
        body: JSON.stringify(data)
      })

      if (!response.ok) {
        const failed = await messageFromFailedResponse(response)
        requestState.error = failed.error
        requestState.requiresVkLink = failed.requiresVkLink
        requestState.vkLinkToken = failed.vkLinkToken
        if (failed.email) {
          requestState.userData.email = failed.email
        }
        return requestState
      }
      
      var responseJSON = await response.json()
      if (responseJSON.error) {
        requestState.error = responseJSON.error
        return requestState
      }

      const ud = responseJSON.userData || {}
      requestState.userData = {
        id: ud.id ?? null,
        firstName: ud.firstName ?? null,
        lastName: ud.lastName ?? null,
        email: ud.email ?? null,
        hasPassword: ud.has_password === true,
      }
      requestState.accessToken = responseJSON.access_token ?? null
      requestState.accessTokenExpiresAt =
        responseJSON.access_token_expires_at ?? null
      requestState.requires_email_verification =
        Boolean(responseJSON.requires_email_verification)
      requestState.emailVerified = ud.email_verified === true
      requestState.verificationResendAfterSeconds = Math.max(
        0,
        Number(responseJSON.verification_resend_after_seconds) || 0
      )
      return requestState
    }

    catch (err) {
      requestState.error = `Ошибка обработки запроса: ${err}`
      return requestState
    }
  }
)
