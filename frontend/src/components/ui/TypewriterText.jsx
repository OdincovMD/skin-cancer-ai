import React, { useEffect, useMemo, useRef, useState } from "react"

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
  const [isAnimating, setIsAnimating] = useState(false)
  const visibleRef = useRef(visible)

  useEffect(() => {
    visibleRef.current = visible
  }, [visible])

  useEffect(() => {
    if (!source || prefersReducedMotion()) {
      setVisible(source)
      setIsAnimating(false)
      return undefined
    }

    const previousVisible = visibleRef.current
    const shouldContinue =
      previousVisible.length > 0 && source.startsWith(previousVisible)

    let index = shouldContinue ? previousVisible.length : 0
    if (!shouldContinue && previousVisible) {
      setVisible("")
    }

    if (index >= source.length) {
      setVisible(source)
      setIsAnimating(false)
      return undefined
    }

    setIsAnimating(true)
    const step = Math.max(1, Math.ceil(source.length / 240))
    const id = window.setInterval(() => {
      index = Math.min(source.length, index + step)
      setVisible(source.slice(0, index))
      if (index >= source.length) {
        setIsAnimating(false)
        window.clearInterval(id)
      }
    }, speed)

    return () => window.clearInterval(id)
  }, [source, speed])

  if (!source) return null

  return (
    <p className={className}>
      {visible}
      {isAnimating && (
        <span
          aria-hidden="true"
          className="ml-0.5 inline-block h-4 w-1 translate-y-0.5 animate-pulse rounded-full bg-med-500"
        />
      )}
    </p>
  )
}

export default TypewriterText
