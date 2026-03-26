import { useEffect, useRef, useState } from "react"

import { bearerAuthHeaders } from "../imports/authHeaders"
import { ME_AVATAR } from "../imports/ENDPOINTS"
import { env } from "../imports/ENV"

export function useAvatarObjectUrl(accessToken, revision = 0) {
  const objectUrlRef = useRef(null)
  const [url, setUrl] = useState(null)

  useEffect(() => {
    if (!accessToken) {
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current)
        objectUrlRef.current = null
      }
      setUrl(null)
      return
    }

    let active = true
    const ctrl = new AbortController()
    const base = env.BACKEND_URL.replace(/\/$/, "")

    ;(async () => {
      try {
        const res = await fetch(`${base}${ME_AVATAR}`, {
          method: "GET",
          cache: "no-store",
          headers: { accept: "image/*", ...bearerAuthHeaders(accessToken) },
          signal: ctrl.signal,
        })
        if (!active) return
        if (res.status === 404 || !res.ok) {
          if (objectUrlRef.current) {
            URL.revokeObjectURL(objectUrlRef.current)
            objectUrlRef.current = null
          }
          setUrl(null)
          return
        }
        const blob = await res.blob()
        if (!active) return
        if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current)
        const next = URL.createObjectURL(blob)
        objectUrlRef.current = next
        setUrl(next)
      } catch (e) {
        if (e?.name === "AbortError" || !active) return
      }
    })()

    return () => {
      active = false
      ctrl.abort()
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current)
        objectUrlRef.current = null
      }
      setUrl(null)
    }
  }, [accessToken, revision])

  return url
}
