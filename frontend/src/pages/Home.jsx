import React, { useEffect, useRef, useState } from "react"
import ReactImageMagnify from "easy-magnify-waft"
import { useSelector } from "react-redux"
import {
  Upload,
  Loader2,
  Trash2,
  Play,
  ScanSearch,
  Gauge,
  History,
  GitBranch,
  AlertTriangle,
  ImageIcon,
  CheckCircle2,
  Blocks,
  FileSearch,
} from "lucide-react"

import { fetchActiveClassificationJob } from "../asyncActions/fetchActiveClassificationJob"
import { handleUploadImage } from "../asyncActions/handleUploadImage"
import {
  clearPendingJob,
  pollClassificationJob,
  savePendingJob,
} from "../asyncActions/pollClassificationJob"
import TreeComponent from "../components/Tree"
import Alert from "../components/ui/Alert"
import BucketLabelsDisclosure, {
  formatFeatureLabelText,
} from "../components/ui/BucketLabelsDisclosure"
import Button from "../components/ui/Button"
import FeatureCard from "../components/ui/FeatureCard"
import TypewriterText from "../components/ui/TypewriterText"
import { env } from "../imports/ENV"
import { HISTORY_IMAGE, PROFILE, SIGN_IN, SIGN_UP } from "../imports/ENDPOINTS"
import { getValues } from "../imports/HELPERS"

function displayNameFromStoredFileName(storedName) {
  if (!storedName) return null
  const m = /^(?:.*?_){3}(?<filename>.*)$/.exec(storedName)
  return m?.groups?.filename ?? storedName
}

const defaultClassificationResult = () => ({
  feature_type: null,
  structure: null,
  properties: [],
  final_class: null,
})

const defaultDescriptionState = () => ({
  status: null,
  text: null,
  error: null,
  importantLabels: [],
  bucketedLabels: [],
})

const Home = () => {
  const userInfo = useSelector((state) => state.user)
  const resumeEffectGen = useRef(0)

  const [isImageLoading, setIsImageLoading] = useState(false)
  const [activeJobLabel, setActiveJobLabel] = useState(null)
  const [fileName, setFileName] = useState(null)
  const [fileData, setFileData] = useState(null)
  const [imageSrc, setImageSrc] = useState(null)
  const [classificationResult, setClassificationResult] = useState(
    defaultClassificationResult
  )
  const [descriptionState, setDescriptionState] = useState(defaultDescriptionState)
  const [isDragging, setIsDragging] = useState(false)
  const [uploadError, setUploadError] = useState(null)

  const isAuthed = Boolean(userInfo.userData?.id && userInfo.accessToken)
  const isVerified = Boolean(userInfo.emailVerified)

  const applyPollingSnapshot = (payload) => {
    if (!payload) return
    setClassificationResult(
      payload.classification ?? defaultClassificationResult()
    )
    setDescriptionState({
      status: payload.descriptionStatus ?? null,
      text: payload.description ?? null,
      error: payload.descriptionError ?? null,
      importantLabels: Array.isArray(payload.importantLabels)
        ? payload.importantLabels
        : [],
      bucketedLabels: Array.isArray(payload.bucketedLabels)
        ? payload.bucketedLabels
        : [],
    })
    if (payload.imageToken) {
      const base = env.BACKEND_URL.replace(/\/$/, "")
      setImageSrc(
        `${base}${HISTORY_IMAGE}?token=${encodeURIComponent(payload.imageToken)}`
      )
    }
    if (
      payload.classification?.final_class ||
      Object.prototype.hasOwnProperty.call(payload.classification ?? {}, "detail")
    ) {
      setIsImageLoading(false)
      setActiveJobLabel(null)
    }
  }

  useEffect(() => {
    const uid = userInfo?.userData?.id
    const token = userInfo?.accessToken
    if (!uid || !token || !userInfo.emailVerified) return
    resumeEffectGen.current += 1
    const gen = resumeEffectGen.current
    let cancelled = false
    let sawProgress = false

    ;(async () => {
      try {
        const active = await fetchActiveClassificationJob(token)
        if (cancelled || resumeEffectGen.current !== gen) return
        if (!active) {
          clearPendingJob(uid)
          setDescriptionState(defaultDescriptionState())
          return
        }
        savePendingJob(uid, active.job_id)
        setActiveJobLabel(displayNameFromStoredFileName(active.file_name))
        setIsImageLoading(true)
        const polled = await pollClassificationJob({
          jobId: active.job_id,
          userId: uid,
          accessToken: token,
          onUpdate: (payload) => {
            if (!cancelled && resumeEffectGen.current === gen) {
              sawProgress = true
              applyPollingSnapshot(payload)
            }
          },
        })
        if (cancelled || resumeEffectGen.current !== gen) return
        applyPollingSnapshot(polled)
      } catch (e) {
        if (!cancelled && resumeEffectGen.current === gen && !sawProgress) {
          setClassificationResult(defaultClassificationResult())
          setDescriptionState(defaultDescriptionState())
        }
      } finally {
        if (!cancelled && resumeEffectGen.current === gen) {
          setIsImageLoading(false)
          setActiveJobLabel(null)
        }
      }
    })()

    return () => {
      cancelled = true
    }
  }, [userInfo?.userData?.id, userInfo?.accessToken, userInfo?.emailVerified])

  const processFile = (file) => {
    if (!file || !file.type.startsWith("image/")) return
    setUploadError(null)
    setFileName(file.name)

    const now = new Date()
    const day = `${now.getDate()}-${now.getMonth() + 1}-${now.getFullYear()}`
    const time = `${now.getHours()}-${now.getMinutes()}-${now.getSeconds()}`
    const stamp = `${day}_${time}_${userInfo.userData.id}_${file.name}`
    const processed = new File([file], stamp, { type: file.type })
    setFileData(processed)

    const reader = new FileReader()
    reader.onload = (e) => setImageSrc(e.target.result)
    reader.readAsDataURL(processed)
    setClassificationResult(defaultClassificationResult())
    setDescriptionState(defaultDescriptionState())
  }

  const handleFileChange = (e) => processFile(e.target.files?.[0])

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    processFile(e.dataTransfer.files[0])
  }

  const handleClassify = () => {
    setUploadError(null)
    setIsImageLoading(true)
    setDescriptionState(defaultDescriptionState())
    handleUploadImage({
      id: userInfo.userData.id,
      fileData,
      accessToken: userInfo.accessToken,
      onProgress: applyPollingSnapshot,
    })
      .then((res) => {
        setUploadError(res.error)
        applyPollingSnapshot(res)
      })
      .catch((err) => {
        setUploadError(String(err?.message || err))
      })
      .finally(() => {
        setIsImageLoading(false)
        setActiveJobLabel(null)
      })
  }

  const resetImage = () => {
    setUploadError(null)
    setClassificationResult(defaultClassificationResult())
    setDescriptionState(defaultDescriptionState())
    setFileName(null)
    setImageSrc(null)
    setFileData(null)
  }

  return (
    <div className="space-y-6">
      <div className="card-elevated">
        <div className="grid gap-8 lg:grid-cols-[minmax(0,1.15fr)_360px] lg:items-stretch">
          <div className="flex flex-col gap-5">
            <div>
              <h1 className="max-w-3xl text-2xl font-bold leading-tight text-gray-900 sm:text-3xl">
                Диагностика новообразований кожи
              </h1>
              <p className="mt-3 max-w-2xl text-base leading-7 text-gray-500">
                Сервис помогает быстро оценить дерматоскопическое изображение,
                выделить визуальные признаки и получить понятное объяснение
                результата на базе машинного обучения и метода Киттлера.
              </p>
            </div>

            <div className="flex items-center gap-3">
              <div className="h-px flex-1 bg-gray-100" />
              <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-gray-400">
                Возможности
              </span>
              <div className="h-px flex-1 bg-gray-100" />
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
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
                title="Интеграция для разработчиков"
                text="Даёт HTTP API для встраивания в свои продукты."
              />
            </div>
          </div>

          <div className="flex flex-col overflow-hidden rounded-xl border border-slate-200 shadow-[0_24px_60px_-36px_rgba(15,23,42,0.4)]">
            <div className="bg-white p-3">
              <div className="relative overflow-hidden rounded-lg bg-med-900">
                <img
                  src="/images/hero-clinical.jpg"
                  alt="Дерматоскопическое изображение для анализа"
                  className="h-72 w-full object-cover lg:h-[300px]"
                />
              </div>
              <div className="px-2 pb-1 pt-3">
                <p className="text-sm font-semibold text-slate-900">
                  Реальное дерматоскопическое изображение
                </p>
                <p className="mt-1 text-xs leading-relaxed text-slate-500">
                  Превью используется как пример входного снимка для анализа.
                </p>
              </div>
            </div>

            <div className="border-t border-amber-100 bg-amber-50 px-4 py-3.5">
              <div className="flex items-start gap-3">
                <AlertTriangle
                  size={16}
                  className="mt-0.5 flex-shrink-0 text-amber-500"
                />
                <div>
                  <p className="text-sm font-semibold text-amber-900">Важно</p>
                  <p className="mt-0.5 text-sm leading-relaxed text-amber-700">
                    Сервис не заменяет консультацию врача. При любых сомнениях
                    обратитесь к квалифицированному дерматологу.
                  </p>
                </div>
              </div>
            </div>

            <div className="border-t border-slate-100 bg-slate-50 px-4 py-3.5">
              <p className="text-xs leading-relaxed text-slate-500">
                Классификаторы обучены на изображениях 2560&times;1920. Другие
                разрешения <em>могут</em> повлиять на точность.
              </p>
            </div>
          </div>

        </div>
      </div>

      {!isAuthed && (
        <div className="card text-center py-10">
          <ImageIcon size={40} className="mx-auto mb-3 text-gray-300" />
          <h2 className="text-lg font-semibold text-gray-800">
            Войдите, чтобы загрузить изображение
          </h2>
          <p className="mt-1 text-sm text-gray-500 max-w-md mx-auto">
            Для использования классификатора необходимо зарегистрироваться и
            подтвердить адрес электронной почты.
          </p>
          <div className="mt-5 flex flex-wrap justify-center gap-3">
            <Button variant="primary" to={SIGN_IN}>
              Войти
            </Button>
            <Button variant="secondary" to={SIGN_UP}>
              Регистрация
            </Button>
          </div>
        </div>
      )}

      {isAuthed && !isVerified && (
        <div className="card border-amber-200 bg-amber-50 text-center py-8">
          <AlertTriangle size={32} className="mx-auto mb-3 text-amber-500" />
          <h2 className="text-lg font-semibold text-amber-900">
            Подтвердите email
          </h2>
          <p className="mt-1 text-sm text-amber-800 max-w-lg mx-auto">
            Загрузка изображений доступна только после подтверждения почты.
            Проверьте входящие и папку «Спам» или запросите письмо повторно.
          </p>
          <Button
            variant="primary"
            to={PROFILE}
            className="mt-4 bg-amber-700 hover:bg-amber-800 focus-visible:ring-amber-500"
          >
            Личный кабинет
          </Button>
        </div>
      )}

      {isAuthed && isVerified && (
        <div className="card-elevated">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            {imageSrc ? "Загруженное изображение" : "Загрузите изображение"}
          </h2>

          {!imageSrc && (
            <label
              onDragOver={(e) => {
                e.preventDefault()
                setIsDragging(true)
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-10 transition-colors
              ${
                isDragging
                  ? "border-med-400 bg-med-50"
                  : "border-gray-300 bg-gray-50 hover:border-med-400 hover:bg-med-50/50"
              }`}
            >
              <Upload
                size={36}
                className={`mb-3 ${
                  isDragging ? "text-med-500" : "text-gray-400"
                }`}
              />
              <span className="text-sm font-medium text-gray-700">
                Перетащите файл сюда или нажмите для выбора
              </span>
              <span className="mt-1 text-xs text-gray-400">
                JPG, PNG, WEBP
              </span>
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleFileChange}
              />
            </label>
          )}

          {imageSrc && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <ImageIcon size={16} className="text-gray-400" />
                <span className="truncate">{fileName}</span>
              </div>

              <div className="flex flex-col items-center gap-4 lg:flex-row lg:items-start">
                <div className="w-full max-w-md">
                  <ReactImageMagnify
                    {...{
                      smallImage: {
                        alt: "Загруженное изображение",
                        isFluidWidth: true,
                        src: imageSrc,
                      },
                      largeImage: {
                        src: imageSrc,
                        width: 2560,
                        height: 1920,
                      },
                      enlargedImagePortalId: "enlargened_image",
                      isHintEnabled: true,
                      shouldHideHintAfterFirstActivation: false,
                      isActivatedOnTouch: true,
                    }}
                  />
                </div>
                <div id="enlargened_image" />
              </div>

              {uploadError && (
                <Alert variant="error" title="Не удалось выполнить загрузку">
                  {uploadError}
                </Alert>
              )}

              {classificationResult.hasOwnProperty("detail") && (
                <Alert variant="error" title="Ошибка обработки">
                  {classificationResult.detail}
                </Alert>
              )}

              <div className="flex flex-wrap gap-3">
                {!isImageLoading &&
                  !classificationResult.final_class &&
                  !classificationResult.hasOwnProperty("detail") && (
                    <Button type="button" onClick={handleClassify}>
                      <Play size={16} />
                      Классифицировать
                    </Button>
                  )}
                {!isImageLoading && (
                  <Button
                    type="button"
                    variant="secondary"
                    onClick={resetImage}
                  >
                    <Trash2 size={16} />
                    Удалить
                  </Button>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {isImageLoading && (
        <div className="card-elevated flex flex-col items-center py-10 text-center">
          <Loader2 size={32} className="animate-spin text-med-500 mb-3" />
          <p className="font-medium text-gray-700">
            Изображение обрабатывается...
          </p>
          {activeJobLabel && (
            <p className="mt-1.5 text-sm text-gray-500 max-w-md break-words">
              {activeJobLabel}
            </p>
          )}
          <p className="mt-3 text-xs text-gray-400">
            Обычно это занимает до 2 минут
          </p>
        </div>
      )}

      {classificationResult.final_class && imageSrc && (
        <div className="card-elevated space-y-4">
          <div className="flex items-center gap-2">
            <CheckCircle2 size={20} className="text-green-500" />
            <h2 className="text-lg font-semibold text-gray-900">
              Результат классификации
            </h2>
          </div>

          <div className="flex flex-wrap gap-2">
            {getValues(classificationResult).map((val, i) => (
              <React.Fragment key={i}>
                {i > 0 && (
                  <span className="self-center text-gray-300">&rarr;</span>
                )}
                <span className="rounded-full bg-med-50 px-3 py-1 text-sm font-medium text-med-800">
                  {val}
                </span>
              </React.Fragment>
            ))}
          </div>

          <TreeComponent
            classificationResult={classificationResult}
            displaySize={{ width: "100%", height: "500px" }}
            nodeSize={{ x: 300, y: 50 }}
            zoom={0.6}
            translate={{ x: 300, y: 300 }}
          />

          {(descriptionState.status ||
            descriptionState.text ||
            descriptionState.error ||
            descriptionState.bucketedLabels.length > 0) && (
            <div className="rounded-xl border border-gray-200 bg-gray-50/70 p-4 space-y-3">
              <div className="flex items-center gap-2">
                <FileText size={18} className="text-med-600" />
                <h3 className="text-base font-semibold text-gray-900">
                  Клиническое описание
                </h3>
              </div>

              {descriptionState.status &&
                descriptionState.status !== "completed" &&
                descriptionState.status !== "error" && (
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <Loader2 size={16} className="animate-spin text-med-500" />
                    <span>Описание формируется и появится автоматически.</span>
                  </div>
              )}

              {descriptionState.text && (
                <TypewriterText
                  text={descriptionState.text}
                  className="text-sm leading-relaxed text-gray-700 whitespace-pre-line"
                />
              )}

              {descriptionState.importantLabels.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {descriptionState.importantLabels.map((label) => (
                    <span
                      key={label}
                      className="rounded-full bg-white px-3 py-1 text-xs font-medium text-gray-600 border border-gray-200"
                    >
                      {formatFeatureLabelText(label)}
                    </span>
                  ))}
                </div>
              )}

              <BucketLabelsDisclosure labels={descriptionState.bucketedLabels} />

              {descriptionState.error && (
                <Alert variant="error" title="Описание недоступно">
                  {descriptionState.error}
                </Alert>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default Home
