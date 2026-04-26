import React, { useEffect } from "react"
import {
  BrainCircuit,
  ExternalLink,
  GitBranch,
  GraduationCap,
  MessageCircle,
  ShieldCheck,
  Sparkles,
} from "lucide-react"

import Button from "../components/ui/Button"

const About = () => {
  useEffect(() => {
    const previousBodyOverflow = document.body.style.overflow
    const previousHtmlOverflow = document.documentElement.style.overflow
    document.body.style.overflow = "hidden"
    document.documentElement.style.overflow = "hidden"

    return () => {
      document.body.style.overflow = previousBodyOverflow
      document.documentElement.style.overflow = previousHtmlOverflow
    }
  }, [])

  return (
    <div className="flex h-[calc(100vh-8.5rem)] overflow-hidden items-center justify-center">
      <section className="w-full overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
        <div className="grid lg:grid-cols-[minmax(0,1fr)_320px]">
          <div className="p-6 sm:p-8">
            <div className="inline-flex items-center gap-2 rounded-full border border-med-100 bg-med-50 px-3 py-1 text-xs font-semibold text-med-700">
              <Sparkles size={14} />
              Skin Cancer AI
            </div>

            <h1 className="mt-5 max-w-2xl text-3xl font-bold leading-tight text-slate-950">
              Сервис ранней диагностики новообразований кожи
            </h1>
            <p className="mt-4 max-w-2xl text-base leading-8 text-slate-600">
              Мы разрабатываем инструмент, который помогает анализировать
              дерматоскопические изображения, выделять ключевые признаки и
              объяснять результат классификации понятным языком.
            </p>

            <div className="mt-7 grid gap-3 sm:grid-cols-3">
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                <BrainCircuit size={20} className="text-med-700" />
                <p className="mt-3 text-sm font-semibold text-slate-950">
                  Машинное обучение
                </p>
                <p className="mt-1 text-xs leading-relaxed text-slate-500">
                  Автоматическая оценка изображения и признаков.
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                <GitBranch size={20} className="text-med-700" />
                <p className="mt-3 text-sm font-semibold text-slate-950">
                  Метод Киттлера
                </p>
                <p className="mt-1 text-xs leading-relaxed text-slate-500">
                  Прозрачная логика диагностического дерева.
                </p>
              </div>
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                <ShieldCheck size={20} className="text-med-700" />
                <p className="mt-3 text-sm font-semibold text-slate-950">
                  Поддержка врача
                </p>
                <p className="mt-1 text-xs leading-relaxed text-slate-500">
                  Сервис помогает, но не заменяет консультацию.
                </p>
              </div>
            </div>
          </div>

          <aside className="border-t border-slate-200 bg-med-50/70 p-6 sm:p-8 lg:border-l lg:border-t-0">
            <div className="flex h-full flex-col justify-between gap-6">
              <div>
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-med-600 text-white shadow-sm">
                  <GraduationCap size={24} />
                </div>
                <p className="mt-4 text-xs font-semibold uppercase tracking-wide text-med-700">
                  Команда проекта
                </p>
                <h2 className="mt-1 text-2xl font-bold text-slate-950">
                  НИЯУ МИФИ
                </h2>
                <p className="mt-3 text-sm leading-7 text-slate-600">
                  Учебно-исследовательский проект на стыке медицинской
                  визуализации, backend-разработки и прикладного ML.
                </p>
              </div>

              <div className="space-y-3">
                <Button
                  variant="secondary"
                  href="https://miro.com/app/board/uXjVMwEeFQ8=/"
                  external
                  className="w-full !justify-between border-med-200 bg-white text-med-800 hover:bg-med-50"
                >
                  <span className="flex items-center gap-2">
                    <GitBranch size={16} />
                    Дерево классификации
                  </span>
                  <ExternalLink size={14} className="text-med-500" />
                </Button>

                <Button
                  variant="secondary"
                  href="https://t.me/horokami"
                  external
                  className="w-full !justify-between border-med-200 bg-white text-med-800 hover:bg-med-50"
                >
                  <span className="flex items-center gap-2">
                    <MessageCircle size={16} />
                    Сообщить о проблеме
                  </span>
                  <ExternalLink size={14} className="text-med-500" />
                </Button>
              </div>
            </div>
          </aside>
        </div>
      </section>
    </div>
  )
}

export default About
