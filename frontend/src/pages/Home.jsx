import React from "react"
import { useSelector } from "react-redux"
import {
  Blocks,
  FileSearch,
  Gauge,
  GitBranch,
  History,
  Microscope,
  ScanSearch,
  ShieldCheck,
} from "lucide-react"

import Button from "../components/ui/Button"
import FeatureCard from "../components/ui/FeatureCard"
import { PROFILE, SIGN_IN, SIGN_UP, WORKSPACE } from "../imports/ENDPOINTS"

const Home = () => {
  const userInfo = useSelector((state) => state.user)
  const isAuthed = Boolean(userInfo.userData?.id && userInfo.accessToken)
  const isVerified = Boolean(userInfo.emailVerified)

  const primaryAction = isAuthed
    ? isVerified
      ? { to: WORKSPACE, label: "Открыть рабочую область" }
      : { to: PROFILE, label: "Подтвердить email" }
    : { to: SIGN_IN, label: "Войти и начать работу" }

  const secondaryAction = isAuthed
    ? { to: PROFILE, label: "Личный кабинет" }
    : { to: SIGN_UP, label: "Регистрация" }

  return (
    <div className="space-y-6">
      <section className="card-elevated overflow-hidden">
        <div className="grid gap-8 lg:grid-cols-[minmax(0,1.1fr)_360px] lg:items-stretch">
          <div className="flex flex-col justify-between gap-6">
            <div>
              <div className="inline-flex items-center gap-2 rounded-full border border-med-100 bg-med-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] text-med-700">
                Skin Cancer AI
              </div>
              <h1 className="mt-4 max-w-3xl text-3xl font-bold leading-tight tracking-tight text-slate-950 sm:text-4xl">
                Удобный сервис для оценки дерматоскопических изображений
              </h1>
              <p className="mt-4 max-w-2xl text-base leading-7 text-slate-600">
                Загрузите снимок, получите результат анализа и понятное
                описание ключевых признаков в спокойном и наглядном формате.
              </p>
            </div>

            <div className="flex flex-wrap gap-3">
              <Button to={primaryAction.to}>
                <ScanSearch size={16} />
                {primaryAction.label}
              </Button>
              <Button variant="secondary" to={secondaryAction.to}>
                {secondaryAction.label}
              </Button>
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              <div className="hover-lift rounded-2xl border border-slate-200 bg-white p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">
                  Рабочая область
                </p>
                <p className="mt-2 text-sm leading-6 text-slate-600">
                  Загрузка изображения, просмотр снимка и анализ собраны в одном
                  удобном месте.
                </p>
              </div>
              <div className="hover-lift rounded-2xl border border-slate-200 bg-white p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">
                  История
                </p>
                <p className="mt-2 text-sm leading-6 text-slate-600">
                  Все предыдущие результаты и описания сохраняются в личном
                  кабинете.
                </p>
              </div>
              <div className="hover-lift rounded-2xl border border-slate-200 bg-white p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">
                  Для разработчиков
                </p>
                <p className="mt-2 text-sm leading-6 text-slate-600">
                  Сервис можно подключить к другим продуктам через отдельную
                  документацию.
                </p>
              </div>
            </div>
          </div>

          <div className="hover-lift flex flex-col overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-[0_24px_60px_-36px_rgba(15,23,42,0.4)]">
            <div className="bg-white p-3">
              <div className="relative overflow-hidden rounded-2xl bg-med-900">
                <img
                  src="/images/hero-clinical.jpg"
                  alt="Дерматоскопическое изображение для анализа"
                  className="h-72 w-full object-cover lg:h-[300px]"
                />
              </div>
            </div>
            <div className="border-t border-slate-100 bg-[linear-gradient(180deg,#ffffff_0%,#f8fafc_100%)] px-5 py-5">
              <p className="text-sm font-semibold text-slate-900">
                Что получает пользователь
              </p>
              <p className="mt-2 text-sm leading-7 text-slate-600">
                После загрузки снимка сервис помогает выделить важные признаки,
                показать ход анализа и собрать результат в понятном виде для
                дальнейшей оценки.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="card-elevated">
        <div className="flex items-center gap-3">
          <div className="h-px flex-1 bg-slate-200" />
          <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
            Возможности платформы
          </span>
          <div className="h-px flex-1 bg-slate-200" />
        </div>

        <div className="mt-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          <FeatureCard
            icon={ScanSearch}
            title="Раннее выявление"
            text="Выделяет подозрительные признаки на раннем этапе."
          />
          <FeatureCard
            icon={Gauge}
            title="Быстрый анализ"
            text="Возвращает предварительный результат за несколько минут."
          />
          <FeatureCard
            icon={History}
            title="Мониторинг"
            text="Сохраняет историю снимков и помогает сравнивать изменения."
          />
          <FeatureCard
            icon={GitBranch}
            title="Прозрачность"
            text="Показывает логику решения шаг за шагом."
          />
          <FeatureCard
            icon={FileSearch}
            title="Описание изображения"
            text="Формирует текстовое описание ключевых находок."
          />
          <FeatureCard
            icon={Blocks}
            title="Интеграция"
            text="Позволяет использовать сервис внутри других продуктов."
          />
        </div>
      </section>

      <section className="card-elevated hover-lift overflow-hidden border-slate-200 bg-[linear-gradient(135deg,#0f172a_0%,#123c3a_100%)] text-white">
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_260px] lg:items-center">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] text-white/80">
              <ShieldCheck size={14} />
              Начать анализ
            </div>
            <h2 className="mt-4 text-2xl font-bold tracking-tight">
              Всё готово для загрузки снимка и получения результата
            </h2>
            <p className="mt-3 max-w-2xl text-sm leading-7 text-white/72">
              Откройте рабочую область, загрузите изображение и получите
              результат с основными признаками, деревом анализа и текстовым
              пояснением.
            </p>
          </div>

          <div className="flex flex-col gap-3 lg:items-end">
            <Button
              to={WORKSPACE}
              className="bg-white text-slate-950 hover:bg-white/90 focus-visible:ring-white/40"
            >
              <ScanSearch size={16} />
              Перейти в рабочую область
            </Button>
            {!isAuthed && (
              <Button
                variant="secondary"
                to={SIGN_UP}
                className="border-white/20 bg-white/10 text-white hover:bg-white/15"
              >
                Создать аккаунт
              </Button>
            )}
          </div>
        </div>
      </section>
    </div>
  )
}

export default Home
