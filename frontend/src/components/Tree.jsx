import React, { useEffect, useMemo, useState } from "react"
import Tree from "react-d3-tree"
import { GitBranch, Minus, Plus, RotateCcw } from "lucide-react"

import { classificationTree } from "../imports/TREE"
import { convertToD3Tree, getValues } from "../imports/HELPERS"

const LABEL_WRAP = 20
const ZOOM_STEP = 0.12
const SCALE_EXTENT = { min: 0.3, max: 1.35 }

function clampZoom(value) {
  return Math.min(SCALE_EXTENT.max, Math.max(SCALE_EXTENT.min, value))
}

function isSameViewport(left, right) {
  if (!left || !right) return false
  return (
    Math.abs(left.zoom - right.zoom) < 0.001 &&
    Math.abs(left.translate.x - right.translate.x) < 0.5 &&
    Math.abs(left.translate.y - right.translate.y) < 0.5
  )
}

function wrapLabel(label) {
  const text = String(label || "").trim()
  if (!text) return [""]

  const words = text.split(/\s+/)
  const lines = []
  let current = ""

  words.forEach((word) => {
    const candidate = current ? `${current} ${word}` : word
    if (candidate.length <= LABEL_WRAP) {
      current = candidate
      return
    }
    if (current) lines.push(current)
    current = word
  })

  if (current) lines.push(current)
  return lines
}

const renderNode = ({ nodeDatum, toggleNode }) => {
  const state = nodeDatum.attributes?.state || "branch"
  const isFinal = Boolean(nodeDatum.attributes?.final)
  const lines = wrapLabel(nodeDatum.name)
  const width = isFinal ? 252 : state === "active" ? 236 : 214
  const lineHeight = 17
  const height = Math.max(58, 28 + lines.length * lineHeight)

  const palette =
    state === "final"
      ? {
          fill: "#f3fbff",
          stroke: "#0891b2",
          text: "#5b6b7f",
          shadow: "rgba(8, 145, 178, 0.12)",
          accent: "#67e8f9",
        }
      : state === "active"
        ? {
            fill: "#f0fdfa",
            stroke: "#14b8a6",
            text: "#6b7a8c",
            shadow: "rgba(20, 184, 166, 0.1)",
            accent: "#2dd4bf",
          }
        : {
            fill: "#f8fafc",
            stroke: "#cbd5e1",
            text: "#475569",
            shadow: "rgba(148, 163, 184, 0.08)",
            accent: "#e2e8f0",
          }

  return (
    <g onClick={toggleNode}>
      <rect
        x={-width / 2 + 4}
        y={-height / 2 + 5}
        rx="18"
        ry="18"
        width={width}
        height={height}
        fill={palette.shadow}
      />
      <rect
        x={-width / 2}
        y={-height / 2}
        rx="18"
        ry="18"
        width={width}
        height={height}
        fill={palette.fill}
        stroke={palette.stroke}
        strokeWidth={isFinal ? 2.1 : state === "active" ? 1.8 : 1.2}
      />
      <rect
        x={-width / 2 + 11}
        y={-height / 2 + 10}
        rx="4"
        ry="4"
        width={state === "branch" ? 0 : 6}
        height={height - 20}
        fill={palette.accent}
      />
      <text
        textAnchor="middle"
        fill={palette.text}
        fontSize="13.2"
        fontWeight={state === "branch" ? 430 : 340}
        letterSpacing="-0.01em"
      >
        {lines.map((line, index) => (
          <tspan
            key={`${nodeDatum.name}-${index}`}
            x="0"
            dy={index === 0 ? `${-(lines.length - 1) * 0.52}em` : "1.12em"}
          >
            {line}
          </tspan>
        ))}
      </text>
    </g>
  )
}

const pathClassFunc = ({ target }) => {
  const state = target.data.attributes?.state
  if (state === "final") return "tree-link tree-link-final"
  if (state === "active") return "tree-link tree-link-active"
  return "tree-link tree-link-muted"
}

const TreeComponent = ({
  classificationResult,
  displaySize,
  nodeSize,
  zoom,
  translate,
}) => {
  const [showBranches, setShowBranches] = useState(true)
  const [viewport, setViewport] = useState(() => ({
    zoom: clampZoom(typeof zoom === "number" ? zoom : 0.6),
    translate: translate ?? { x: 0, y: 0 },
  }))
  const startingPoint = "Начало"
  const values = useMemo(
    () => [startingPoint, ...getValues(classificationResult)],
    [classificationResult]
  )
  const initialViewport = useMemo(
    () => ({
      zoom: clampZoom(typeof zoom === "number" ? zoom : 0.6),
      translate: translate ?? { x: 0, y: 0 },
    }),
    [translate?.x, translate?.y, zoom]
  )

  const routeLabels = useMemo(() => values.slice(1), [values])

  const treeData = useMemo(
    () =>
      convertToD3Tree({
        node: classificationTree,
        reference: values,
        index: 0,
        depth: 0,
        includeBranches: showBranches,
      }),
    [showBranches, values]
  )

  const finalLabel =
    routeLabels.length > 0 ? routeLabels[routeLabels.length - 1] : null

  useEffect(() => {
    setViewport((current) =>
      isSameViewport(current, initialViewport) ? current : initialViewport
    )
  }, [initialViewport])

  const changeZoom = (delta) => {
    setViewport((current) => ({
      ...current,
      zoom: clampZoom(current.zoom + delta),
    }))
  }

  const resetViewport = () => {
    setViewport(initialViewport)
  }

  return (
    <div
      className="overflow-hidden rounded-2xl border border-slate-200 bg-[radial-gradient(circle_at_top,#f0fdfa,transparent_38%),linear-gradient(180deg,#ffffff_0%,#f8fafc_100%)] shadow-[0_18px_50px_-40px_rgba(15,23,42,0.45)]"
      style={displaySize}
    >
      <div className="border-b border-slate-200 bg-white/80 px-4 py-3 backdrop-blur">
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span className="inline-flex items-center gap-1.5 rounded-full bg-med-50 px-3 py-1 font-semibold text-med-800 ring-1 ring-med-100">
            <GitBranch size={14} />
            Ход анализа
          </span>
          {routeLabels.slice(0, -1).map((label, index) => (
            <React.Fragment key={`${label}-${index}`}>
              <span className="text-slate-300">/</span>
              <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-700">
                {label}
              </span>
            </React.Fragment>
          ))}
          {finalLabel && (
            <>
              <span className="text-slate-300">/</span>
              <span className="rounded-full bg-cyan-50 px-3 py-1 text-xs font-semibold text-cyan-800 ring-1 ring-cyan-100">
                {finalLabel}
              </span>
            </>
          )}
        </div>
        <div className="mt-2 flex flex-wrap items-start justify-between gap-3">
          <div className="flex flex-wrap gap-2 text-[11px] text-slate-500">
            <button
              type="button"
              onClick={() => setShowBranches((current) => !current)}
              className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-3 py-1 font-medium text-slate-600 transition-colors hover:border-slate-300 hover:bg-slate-50"
            >
              {showBranches ? "Показать только основной путь" : "Показать все ветки"}
            </button>
            <span className="inline-flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-teal-500" />
              активный путь
            </span>
            {showBranches && (
              <span className="inline-flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-slate-300" />
                соседние ветки
              </span>
            )}
            <span className="inline-flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-cyan-500" />
              итоговый узел
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] font-medium text-slate-600">
              Масштаб {Math.round(viewport.zoom * 100)}%
            </span>
            <div className="inline-flex items-center rounded-full border border-slate-200 bg-white p-1 shadow-sm">
              <button
                type="button"
                onClick={() => changeZoom(ZOOM_STEP)}
                className="inline-flex h-8 w-8 items-center justify-center rounded-full text-slate-600 transition-colors hover:bg-slate-100"
                aria-label="Увеличить дерево"
              >
                <Plus size={15} />
              </button>
              <button
                type="button"
                onClick={() => changeZoom(-ZOOM_STEP)}
                className="inline-flex h-8 w-8 items-center justify-center rounded-full text-slate-600 transition-colors hover:bg-slate-100"
                aria-label="Уменьшить дерево"
              >
                <Minus size={15} />
              </button>
              <button
                type="button"
                onClick={resetViewport}
                className="inline-flex h-8 items-center justify-center rounded-full px-2.5 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-100"
                aria-label="Сбросить масштаб и центрировать дерево"
              >
                <RotateCcw size={14} className="mr-1" />
                Сброс
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="relative h-full w-full">
        <style>
          {`
            .tree-link {
              fill: none;
              stroke-linecap: round;
              stroke-linejoin: round;
              transition: stroke 160ms ease, stroke-width 160ms ease;
            }
            .tree-link-active {
              stroke: #14b8a6;
              stroke-width: 2.4px;
            }
            .tree-link-final {
              stroke: #0891b2;
              stroke-width: 2.7px;
            }
            .tree-link-muted {
              stroke: #cbd5e1;
              stroke-width: 1.4px;
              stroke-dasharray: 5 6;
            }
          `}
        </style>
        <Tree
          data={treeData}
          zoom={viewport.zoom}
          translate={viewport.translate}
          orientation="horizontal"
          nodeSize={nodeSize}
          separation={{ siblings: 1.55, nonSiblings: 1.9 }}
          scaleExtent={SCALE_EXTENT}
          onUpdate={(nextViewport) => {
            const normalizedViewport = {
              zoom: clampZoom(nextViewport.zoom),
              translate: nextViewport.translate,
            }
            setViewport((current) =>
              isSameViewport(current, normalizedViewport)
                ? current
                : normalizedViewport
            )
          }}
          renderCustomNodeElement={renderNode}
          pathClassFunc={pathClassFunc}
        />
      </div>
    </div>
  )
}

export default TreeComponent
