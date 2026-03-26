import React from "react"

const Spinner = ({ className = "", size = "md" }) => {
  const dim = size === "sm" ? "h-4 w-4 border-2" : "h-8 w-8 border-2"
  return (
    <span
      className={`inline-block animate-spin rounded-full border-med-500 border-t-transparent ${dim} ${className}`.trim()}
      role="status"
      aria-label="Загрузка"
    />
  )
}

export default Spinner
