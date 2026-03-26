import React from "react"

const FeatureCard = ({ icon: Icon, title, text }) => (
  <div className="flex gap-3">
    <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-lg bg-med-50 text-med-600">
      <Icon size={18} aria-hidden />
    </div>
    <div>
      <h3 className="text-sm font-semibold text-gray-900">{title}</h3>
      <p className="text-sm text-gray-500 leading-relaxed">{text}</p>
    </div>
  </div>
)

export default FeatureCard
