import React, { useMemo, useState } from "react"
import { ChevronDown, ListChecks } from "lucide-react"

const CATEGORY_META = {
  shape: {
    title: "Форма и размер",
    dot: "bg-emerald-500",
    panel: "border-emerald-100 bg-emerald-50/45",
  },
  border: {
    title: "Границы и контур",
    dot: "bg-amber-500",
    panel: "border-amber-100 bg-amber-50/45",
  },
  color: {
    title: "Цветовой паттерн",
    dot: "bg-rose-500",
    panel: "border-rose-100 bg-rose-50/45",
  },
  texture: {
    title: "Текстура",
    dot: "bg-sky-500",
    panel: "border-sky-100 bg-sky-50/45",
  },
  other: {
    title: "Прочее",
    dot: "bg-gray-400",
    panel: "border-gray-100 bg-gray-50",
  },
}

const CATEGORY_ORDER = ["shape", "border", "color", "texture", "other"]

const FEATURE_META = {
  shape: ["shape", "Форма образования"],
  elongation: ["shape", "Вытянутость"],
  eccentricity: ["shape", "Эксцентриситет"],
  area: ["shape", "Площадь маски"],
  perimeter: ["shape", "Периметр маски"],
  circularity: ["shape", "Круглость"],
  aspect_ratio: ["shape", "Соотношение сторон"],
  solidity: ["shape", "Плотность к выпуклой оболочке"],
  extent: ["shape", "Заполнение ограничивающей области"],

  borders: ["border", "Характер границ"],
  asymmetry: ["border", "Симметрия образования"],
  rim: ["border", "Краевой ободок"],
  lobulation: ["border", "Лобуляция"],
  convexity: ["border", "Выпуклость контура"],
  radial_variance: ["border", "Вариация радиуса до контура"],
  perimeter_area_ratio: ["border", "Периметр / sqrt(площади)"],
  fractal_dimension: ["border", "Изрезанность контура"],

  dominant_hue: ["color", "Преобладающий оттенок"],
  pigmentation: ["color", "Пигментация"],
  color_homogeneity: ["color", "Однородность окраски"],
  contrast: ["color", "Контраст с окружающей кожей"],
  palette: ["color", "Цветовая палитра"],
  mean_H_lesion: ["color", "Средний оттенок H в очаге"],
  mean_S_lesion: ["color", "Средняя насыщенность S в очаге"],
  mean_V_lesion: ["color", "Средняя яркость V в очаге"],
  std_H_lesion: ["color", "Разброс оттенка H"],
  std_S_lesion: ["color", "Разброс насыщенности S"],
  std_V_lesion: ["color", "Разброс яркости V"],
  entropy_H_lesion: ["color", "Энтропия канала H"],
  entropy_S_lesion: ["color", "Энтропия канала S"],
  entropy_V_lesion: ["color", "Энтропия канала V"],
  color_balance_R: ["color", "Доля красного канала"],
  color_balance_G: ["color", "Доля зелёного канала"],
  color_balance_B: ["color", "Доля синего канала"],
  color_distance_euclidean: ["color", "LAB-расстояние очаг↔кожа"],
  color_distance_deltaE: ["color", "ΔE2000 очаг↔кожа"],
  dominant_colors_lesion: ["color", "Топ доминантных цветов очага"],
  percent_dark_pixels: ["color", "Доля очень тёмных пикселей"],
  percent_white_pixels: ["color", "Доля белых пикселей"],
  percent_red_pixels: ["color", "Доля красных пикселей"],
  percent_blue_pixels: ["color", "Доля синих пикселей"],
  percent_outlier_bright_pixels: ["color", "Доля аномально ярких пикселей"],
  percent_outlier_dark_pixels: ["color", "Доля аномально тёмных пикселей"],
  delta_H_center_periphery: ["color", "Δ оттенка центр↔периферия"],
  delta_S_center_periphery: ["color", "Δ насыщенности центр↔периферия"],
  delta_V_center_periphery: ["color", "Δ яркости центр↔периферия"],
  delta_V_left_right: ["color", "Δ яркости левая↔правая половины"],
  delta_S_left_right: ["color", "Δ насыщенности левая↔правая половины"],
  delta_V_top_bottom: ["color", "Δ яркости верх↔низ"],
  delta_S_top_bottom: ["color", "Δ насыщенности верх↔низ"],
  delta_V_inner_rim: ["color", "Δ яркости центр↔краевая зона"],

  texture: ["texture", "Текстура поверхности"],
  texture_coarseness: ["texture", "Грубость текстуры"],
  structure_order: ["texture", "Упорядоченность структуры"],
  glcm_contrast: ["texture", "GLCM-контраст"],
  glcm_homogeneity: ["texture", "GLCM-однородность"],
  glcm_energy: ["texture", "GLCM-энергия"],
  glcm_entropy: ["texture", "GLCM-энтропия"],
  lbp_uniformity: ["texture", "LBP-равномерность"],
  lbp_entropy: ["texture", "LBP-энтропия"],
  lbp_mean: ["texture", "LBP-среднее"],
  lbp_std: ["texture", "LBP-стандартное отклонение"],
  lbp_median: ["texture", "LBP-медиана"],
}

function humanizeToken(value) {
  return String(value ?? "")
    .replace(/^bucket_/, "")
    .replaceAll("_", " ")
    .replace(/\s+/g, " ")
    .trim()
}

function capitalizeFirst(value) {
  const text = String(value ?? "").trim()
  if (!text) return ""
  return text.charAt(0).toUpperCase() + text.slice(1)
}

function normalizeValue(value) {
  const raw = String(value ?? "").trim()
  if (!raw) return ""
  const cleanValue = raw.includes(":") ? raw.split(":").pop() : raw
  return capitalizeFirst(humanizeToken(cleanValue))
}

function normalizeKey(key) {
  return String(key ?? "").trim().replace(/^bucket_/, "")
}

function parseBucketLabel(label) {
  if (typeof label !== "string") {
    return {
      category: "other",
      rawKey: "",
      title: "Нераспознанный бакет",
      value: String(label ?? ""),
    }
  }
  const index = label.indexOf(":")
  const rawKey = normalizeKey(index === -1 ? "" : label.slice(0, index))
  const rawValue = index === -1 ? label : label.slice(index + 1)
  const meta = FEATURE_META[rawKey]

  if (index === -1) {
    return {
      category: "other",
      rawKey: "",
      title: "Бакет",
      value: normalizeValue(label),
    }
  }

  return {
    category: meta?.[0] ?? "other",
    rawKey,
    title: meta?.[1] ?? capitalizeFirst(humanizeToken(rawKey)),
    value: normalizeValue(rawValue),
  }
}

export function formatFeatureLabelText(label) {
  const entry = parseBucketLabel(label)
  if (!entry.title || entry.title === "Бакет") {
    return entry.value
  }
  return `${entry.title}: ${entry.value}`
}

function groupEntries(entries) {
  return CATEGORY_ORDER.map((category) => ({
    category,
    items: entries.filter((entry) => entry.category === category),
  })).filter((group) => group.items.length > 0)
}

const BucketLabelsDisclosure = ({ labels, className = "" }) => {
  const [open, setOpen] = useState(false)
  const entries = useMemo(
    () => (Array.isArray(labels) ? labels.map(parseBucketLabel) : []),
    [labels]
  )
  const groups = useMemo(() => groupEntries(entries), [entries])

  if (entries.length === 0) return null

  return (
    <div
      className={`overflow-hidden rounded-lg border border-gray-200 bg-white shadow-sm ${className}`}
    >
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center justify-between gap-3 px-3 py-2.5 text-left text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
        aria-expanded={open}
      >
        <span className="inline-flex min-w-0 items-center gap-2">
          <ListChecks size={15} className="shrink-0 text-med-600" />
          <span className="truncate">Все признаки ({entries.length})</span>
        </span>
        <ChevronDown
          size={16}
          className={`shrink-0 text-gray-400 transition-transform ${
            open ? "rotate-180" : ""
          }`}
        />
      </button>

      {open && (
        <div className="space-y-3 border-t border-gray-100 bg-gray-50/60 px-3 py-3">
          {groups.map((group) => {
            const meta = CATEGORY_META[group.category] ?? CATEGORY_META.other
            return (
              <section
                key={group.category}
                className={`rounded-lg border px-3 py-3 ${meta.panel}`}
              >
                <div className="mb-2 flex items-center gap-2">
                  <span className={`h-2 w-2 rounded-full ${meta.dot}`} />
                  <h4 className="text-xs font-semibold uppercase tracking-normal text-gray-600">
                    {meta.title}
                  </h4>
                </div>
                <div className="grid gap-2 md:grid-cols-2">
                  {group.items.map((entry, index) => (
                    <div
                      key={`${entry.rawKey || entry.title}_${entry.value}_${index}`}
                      className="min-w-0 rounded-md border border-white/70 bg-white/85 px-2.5 py-2 shadow-sm"
                    >
                      <div className="text-[11px] font-medium leading-snug text-gray-500">
                        {entry.title}
                      </div>
                      <div className="mt-1 break-words text-xs font-semibold leading-snug text-gray-800">
                        {entry.value}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )
          })}
        </div>
      )}
    </div>
  )
}

export default BucketLabelsDisclosure
