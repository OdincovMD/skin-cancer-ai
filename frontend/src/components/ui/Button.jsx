import React from "react"
import { Link } from "react-router-dom"

const variantClass = {
  primary: "btn-primary",
  secondary: "btn-secondary",
  danger: "btn-danger",
}

/**
 * @param {object} props
 * @param {"primary"|"secondary"|"danger"} [props.variant]
 * @param {string} [props.to] — internal route (React Router)
 * @param {string} [props.href] — external URL
 * @param {boolean} [props.external] — open href in new tab
 */
const Button = ({
  variant = "primary",
  to,
  href,
  external,
  className = "",
  disabled,
  type = "button",
  children,
  ...rest
}) => {
  const base = variantClass[variant] || variantClass.primary
  const combined = `${base} ${className}`.trim()

  if (href) {
    return (
      <a
        href={href}
        className={combined}
        target={external ? "_blank" : undefined}
        rel={external ? "noopener noreferrer" : undefined}
        {...rest}
      >
        {children}
      </a>
    )
  }

  if (to) {
    return (
      <Link
        to={to}
        className={
          disabled
            ? `${combined} pointer-events-none opacity-50 cursor-not-allowed`
            : combined
        }
        aria-disabled={disabled || undefined}
        {...rest}
      >
        {children}
      </Link>
    )
  }

  return (
    <button type={type} disabled={disabled} className={combined} {...rest}>
      {children}
    </button>
  )
}

export default Button
