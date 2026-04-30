import { env } from "../imports/ENV"
import { createAsyncThunk } from "@reduxjs/toolkit"

async function messageFromFailedResponse(response) {
  try {
    const body = await response.json()
    const d = body?.detail
    if (d != null) {
      if (typeof d === "string") return d
      if (Array.isArray(d) && d[0]?.msg != null) return String(d[0].msg)
      return JSON.stringify(d)
    }
    if (body?.error != null && typeof body.error === "string") {
      return body.error
    }
  } catch {
  }
  const s = response.status
  if (s === 401) return "Неверный логин или пароль."
  if (s === 403) return "Доступ запрещён."
  if (s === 422) return "Проверьте введённые данные."
  if (s === 503) return "Сервис временно недоступен. Попробуйте позже."
  if (s >= 500) return "Ошибка сервера. Попробуйте позже."
  return `Ошибка запроса (${s}). Повторите попытку.`
}

export const onVerify = createAsyncThunk("user/onVerify", async ({data, endpoint}) => {

    let requestState = {
      userData: {
        id: null,
        firstName: null,
        lastName: null,
        email: null,
      },
      accessToken: null,
      accessTokenExpiresAt: null,
      error: null,
      requires_email_verification: false,
      emailVerified: true,
      verificationResendAfterSeconds: 0,
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
        requestState.error = await messageFromFailedResponse(response)
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
