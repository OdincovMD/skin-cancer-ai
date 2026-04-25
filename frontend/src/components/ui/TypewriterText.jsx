import React, { useEffect, useMemo, useState } from "react"

function prefersReducedMotion() {
  return (
    typeof window !== "undefined" &&
    window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches
  )
}

const TypewriterText = ({ text, className = "", speed = 12 }) => {
  const source = useMemo(() => String(text ?? ""), [text])
  const [visible, setVisible] = useState(() =>
    prefersReducedMotion() ? source : ""
  )

  useEffect(() => {
    if (!source || prefersReducedMotion()) {
      setVisible(source)
      return undefined
    }

    setVisible("")
    let index = 0
    const step = Math.max(1, Math.ceil(source.length / 240))
    const id = window.setInterval(() => {
      index = Math.min(source.length, index + step)
      setVisible(source.slice(0, index))
      if (index >= source.length) {
        window.clearInterval(id)
      }
    }, speed)

    return () => window.clearInterval(id)
  }, [source, speed])

  if (!source) return null

  const done = visible.length >= source.length

  return (
    <p className={className}>
      {visible}
      {!done && (
        <span
          aria-hidden="true"
          className="ml-0.5 inline-block h-4 w-1 translate-y-0.5 animate-pulse rounded-full bg-med-500"
        />
      )}
    </p>
  )
}

export default TypewriterText
