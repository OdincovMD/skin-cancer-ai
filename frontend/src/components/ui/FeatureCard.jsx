import React from "react"

const FeatureCard = ({ icon: Icon, title, text }) => (
  <div className="flex gap-4 rounded-xl border border-slate-200/90 bg-white px-4 py-4 shadow-[0_18px_40px_-34px_rgba(15,23,42,0.45)] transition-transform duration-200 hover:-translate-y-0.5">
    <div className="flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-lg border border-teal-100 bg-gradient-to-br from-teal-50 via-cyan-50 to-white text-teal-700 shadow-inner">
      <Icon size={18} strokeWidth={2.1} aria-hidden />
    </div>
    <div className="min-w-0">
      <h3 className="text-sm font-semibold tracking-tight text-slate-900">
        {title}
      </h3>
      <p className="mt-1 text-sm leading-relaxed text-slate-600">{text}</p>
    </div>
  </div>
)

export default FeatureCard
