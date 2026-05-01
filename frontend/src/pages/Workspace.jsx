import React from "react"
import { ArrowRight, Microscope, ScanSearch } from "lucide-react"

import ClassificationWorkspace from "../components/ClassificationWorkspace"

const Workspace = () => {
  return (
    <div className="space-y-6">
      <section className="card-elevated overflow-hidden">
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1.15fr)_320px] lg:items-center">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-med-100 bg-med-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] text-med-700">
              Рабочая область
            </div>
            <h1 className="mt-4 max-w-3xl text-3xl font-bold tracking-tight text-slate-950 sm:text-4xl">
              Всё для анализа снимка в одном месте
            </h1>
            <p className="mt-4 max-w-2xl text-base leading-7 text-slate-600">
              Здесь можно загрузить изображение, внимательно рассмотреть его,
              выбрать подходящий режим и получить результат в удобной форме.
            </p>
          </div>

          <div className="hover-lift rounded-3xl border border-slate-200 bg-[linear-gradient(180deg,#ffffff_0%,#f8fafc_100%)] p-5 shadow-[0_20px_40px_-34px_rgba(15,23,42,0.35)]">
            <div className="flex items-start gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-med-600 text-white shadow-sm">
                <ScanSearch size={20} />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-900">
                  Что внутри
                </p>
                <ul className="mt-3 space-y-2 text-sm leading-6 text-slate-600">
                  <li className="flex items-center gap-2">
                    <ArrowRight size={14} className="text-med-600" />
                    загрузка изображения и выбор режима
                  </li>
                  <li className="flex items-center gap-2">
                    <ArrowRight size={14} className="text-med-600" />
                    просмотр снимка и этапов обработки
                  </li>
                  <li className="flex items-center gap-2">
                    <ArrowRight size={14} className="text-med-600" />
                    результат, дерево анализа и описание
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      <ClassificationWorkspace />
    </div>
  )
}

export default Workspace
