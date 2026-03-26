import React from "react"
import { AlertCircle, AlertTriangle, CheckCircle2, Info } from "lucide-react"

const styles = {
  error: {
    wrap: "bg-red-50 border-red-200 text-red-800",
    icon: AlertCircle,
    iconClass: "text-red-500",
  },
  warning: {
    wrap: "bg-amber-50 border-amber-200 text-amber-900",
    icon: AlertTriangle,
    iconClass: "text-amber-600",
  },
  success: {
    wrap: "bg-green-50 border-green-200 text-green-800",
    icon: CheckCircle2,
    iconClass: "text-green-600",
  },
  info: {
    wrap: "bg-med-50 border-med-200 text-med-900",
    icon: Info,
    iconClass: "text-med-600",
  },
}

/**
 * @param {object} props
 * @param {"error"|"warning"|"success"|"info"} [props.variant]
 * @param {string} [props.title]
 * @param {boolean} [props.icon] — show icon (default true)
 */
const Alert = ({ variant = "error", title, children, icon = true, className = "" }) => {
  const cfg = styles[variant] || styles.error
  const Icon = cfg.icon
  return (
    <div
      role="alert"
      className={`flex gap-3 rounded-lg border px-4 py-3 text-sm leading-relaxed ${cfg.wrap} ${className}`.trim()}
    >
      {icon && (
        <Icon
          size={18}
          className={`mt-0.5 flex-shrink-0 ${cfg.iconClass}`}
          aria-hidden
        />
      )}
      <div className="min-w-0 flex-1">
        {title && <p className="font-semibold mb-0.5">{title}</p>}
        <div>{children}</div>
      </div>
    </div>
  )
}

export default Alert
