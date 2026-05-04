import React, { useEffect, useRef, useState } from "react"
import ReactImageMagnify from "easy-magnify-waft"
import { useSelector } from "react-redux"
import {
  AlertTriangle,
  CheckCircle2,
  FileText,
  ImageIcon,
  Loader2,
  Play,
  ScanSearch,
  Trash2,
  Upload,
} from "lucide-react"

import { fetchActiveClassificationJob } from "../asyncActions/fetchActiveClassificationJob"
import { handleUploadImage } from "../asyncActions/handleUploadImage"
import {
  clearPendingJob,
  pollClassificationJob,
  savePendingJob,
} from "../asyncActions/pollClassificationJob"
import { env } from "../imports/ENV"
import {
  HISTORY_IMAGE,
  PROFILE,
  SIGN_IN,
  SIGN_UP,
} from "../imports/ENDPOINTS"
import { getValues } from "../imports/HELPERS"
import TreeComponent from "./Tree"
import Alert from "./ui/Alert"
import BucketLabelsDisclosure, {
  formatFeatureLabelText,
} from "./ui/BucketLabelsDisclosure"
import Button from "./ui/Button"
import TypewriterText from "./ui/TypewriterText"

function displayNameFromStoredFileName(storedName) {
  if (!storedName) return null
  const baseName = storedName.split("/").pop()
  const withoutHex = /^[0-9a-fA-F]{16}_(?<filename>.*)$/.exec(baseName)?.groups
    ?.filename ?? baseName
  const stampedMatch =
    /^(?:\d{1,2}-){2}\d{4}_(?:\d{1,2}-){2}\d{1,2}_\d+_(?<filename>.*)$/.exec(
      withoutHex
    )
  return stampedMatch?.groups?.filename ?? withoutHex
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

const defaultStageState = () => ({
  key: null,
  title: null,
  description: null,
})

const analysisSteps = [
  { key: "preparing", label: "Подготовка" },
  { key: "mask", label: "Маска" },
  { key: "classification", label: "Анализ" },
  { key: "finalizing", label: "Результат" },
]

const analysisModes = {
  full: {
    value: "full",
    label: "Полный анализ",
    description: "Классификация и клиническое описание",
    featuresOnly: false,
  },
  features: {
    value: "features",
    label: "Только классификация",
    description: "Без текстового описания",
    featuresOnly: true,
  },
}

const ClassificationWorkspace = () => {
  const userInfo = useSelector((state) => state.user)
  const resumeEffectGen = useRef(0)
  const resultsRef = useRef(null)
  const imageContainerRef = useRef(null)

  const [isImageLoading, setIsImageLoading] = useState(false)
  const [activeJobLabel, setActiveJobLabel] = useState(null)
  const [fileName, setFileName] = useState(null)
  const [fileData, setFileData] = useState(null)
  const [imageSrc, setImageSrc] = useState(null)
  const [classificationResult, setClassificationResult] = useState(
    defaultClassificationResult
  )
  const [descriptionState, setDescriptionState] = useState(defaultDescriptionState)
  const [analysisStage, setAnalysisStage] = useState(defaultStageState)
  const [analysisMode, setAnalysisMode] = useState(analysisModes.full.value)
  const [isDragging, setIsDragging] = useState(false)
  const [uploadError, setUploadError] = useState(null)

  const isAuthed = Boolean(userInfo.userData?.id && userInfo.accessToken)
  const isVerified = Boolean(userInfo.emailVerified)

  const applyPollingSnapshot = (payload) => {
    if (!payload) return
    setClassificationResult(
      payload.classification ?? defaultClassificationResult()
    )
    setDescriptionState((prev) => {
      const nextText =
        typeof payload.description === "string" && payload.description.length > 0
          ? payload.description
          : prev.text
      const nextImportantLabels = Array.isArray(payload.importantLabels)
        ? payload.importantLabels.length > 0
          ? payload.importantLabels
          : prev.importantLabels
        : prev.importantLabels
      const nextBucketedLabels = Array.isArray(payload.bucketedLabels)
        ? payload.bucketedLabels.length > 0
          ? payload.bucketedLabels
          : prev.bucketedLabels
        : prev.bucketedLabels

      return {
        status: payload.descriptionStatus ?? prev.status ?? null,
        text: nextText,
        error: payload.descriptionError ?? prev.error ?? null,
        importantLabels: nextImportantLabels,
        bucketedLabels: nextBucketedLabels,
      }
    })
    setAnalysisStage(payload.stage ?? defaultStageState())
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
      setAnalysisStage(defaultStageState())
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
          setFileName(null)
          setDescriptionState(defaultDescriptionState())
          setAnalysisStage(defaultStageState())
          return
        }
        savePendingJob(uid, active.job_id)
        const readableFileName = displayNameFromStoredFileName(active.file_name)
        setActiveJobLabel(readableFileName)
        setFileName(readableFileName)
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
          setAnalysisStage(defaultStageState())
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

  useEffect(() => {
    if (classificationResult.final_class && imageSrc && resultsRef.current) {
      resultsRef.current.scrollIntoView({ behavior: "smooth", block: "start" })
    } else if (
      imageSrc &&
      !classificationResult.final_class &&
      imageContainerRef.current
    ) {
      imageContainerRef.current.scrollIntoView({ behavior: "smooth", block: "start" })
    }
  }, [classificationResult.final_class, imageSrc])

  const processFile = (file) => {
    if (!file || !file.type.startsWith("image/")) return
    setUploadError(null)
    setFileName(displayNameFromStoredFileName(file.name))

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
    setAnalysisStage(defaultStageState())
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
    const selectedMode = analysisModes[analysisMode] || analysisModes.full
    setAnalysisStage({
      key: "preparing",
      title: "Подготовка изображения",
      description: "Загружаем файл и запускаем обработку.",
    })
    handleUploadImage({
      id: userInfo.userData.id,
      fileData,
      accessToken: userInfo.accessToken,
      featuresOnly: selectedMode.featuresOnly,
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
        setAnalysisStage(defaultStageState())
      })
  }

  const resetImage = () => {
    setUploadError(null)
    setClassificationResult(defaultClassificationResult())
    setDescriptionState(defaultDescriptionState())
    setAnalysisStage(defaultStageState())
    setFileName(null)
    setImageSrc(null)
    setFileData(null)
  }

  const selectedMode = analysisModes[analysisMode] || analysisModes.full
  const classifyButtonLabel = selectedMode.featuresOnly
    ? "Классифицировать"
    : "Выполнить полный анализ"
  const hasDescriptionLabels =
    descriptionState.importantLabels.length > 0 ||
    descriptionState.bucketedLabels.length > 0
  const shouldShowMissingDescriptionFallback =
    descriptionState.status === "completed" &&
    !descriptionState.text &&
    !descriptionState.error &&
    hasDescriptionLabels

  return (
    <div className="space-y-6">
      {!isAuthed && (
        <div className="card text-center py-10">
          <ImageIcon size={40} className="mx-auto mb-3 text-gray-300" />
          <h2 className="text-lg font-semibold text-gray-800">
            Войдите, чтобы открыть рабочую область
          </h2>
          <p className="mt-1 text-sm text-gray-500 max-w-md mx-auto">
            Для загрузки изображения и получения результатов необходимо
            зарегистрироваться и подтвердить адрес электронной почты.
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
        <div className="card-elevated" ref={imageContainerRef}>
          <h2 className="mb-4 text-lg font-semibold text-gray-900">
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
              className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-10 transition-colors ${
                isDragging
                  ? "border-med-400 bg-med-50"
                  : "border-gray-300 bg-gray-50 hover:border-med-400 hover:bg-med-50/50"
              }`}
            >
              <Upload
                size={36}
                className={`mb-3 ${isDragging ? "text-med-500" : "text-gray-400"}`}
              />
              <span className="text-sm font-medium text-gray-700">
                Перетащите файл сюда или нажмите для выбора
              </span>
              <span className="mt-1 text-xs text-gray-400">JPG, PNG, WEBP</span>
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

              <div className="space-y-3">
                {!isImageLoading &&
                  !classificationResult.final_class &&
                  !classificationResult.hasOwnProperty("detail") && (
                    <div className="overflow-hidden rounded-2xl border border-slate-200 bg-[linear-gradient(180deg,#f8fafc_0%,#ffffff_100%)] shadow-[0_18px_40px_-34px_rgba(15,23,42,0.35)]">
                      <div className="flex flex-col gap-3 border-b border-slate-200 bg-white/90 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
                        <div>
                          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                            Режим работы
                          </p>
                          <p className="mt-1 text-sm text-slate-600">
                            Выберите формат результата перед запуском анализа.
                          </p>
                        </div>
                        <div className="inline-flex items-center self-start rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-600">
                          {selectedMode.featuresOnly
                            ? "Только признаки"
                            : "Признаки и описание"}
                        </div>
                      </div>

                      <div className="grid gap-3 p-3 md:grid-cols-2">
                        {Object.values(analysisModes).map((mode) => {
                          const selected = analysisMode === mode.value
                          const Icon = mode.featuresOnly ? ScanSearch : FileText
                          return (
                            <button
                              key={mode.value}
                              type="button"
                              onClick={() => setAnalysisMode(mode.value)}
                              className={`group relative overflow-hidden rounded-2xl border px-4 py-4 text-left transition-all duration-200 ${
                                selected
                                  ? "border-med-300 bg-[linear-gradient(135deg,#ecfeff_0%,#f0fdfa_100%)] shadow-[0_18px_30px_-26px_rgba(13,148,136,0.45)] ring-1 ring-med-100"
                                  : "border-slate-200 bg-white hover:-translate-y-0.5 hover:border-slate-300 hover:shadow-[0_18px_30px_-28px_rgba(15,23,42,0.24)]"
                              }`}
                            >
                              <div className="flex items-start gap-3">
                                <span
                                  className={`mt-0.5 inline-flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-2xl border ${
                                    selected
                                      ? "border-med-200 bg-white text-med-700"
                                      : "border-slate-200 bg-slate-50 text-slate-500 group-hover:text-slate-700"
                                  }`}
                                >
                                  <Icon size={18} />
                                </span>
                                <div className="min-w-0 flex-1">
                                  <div className="flex items-center justify-between gap-3">
                                    <p className="text-sm font-semibold text-slate-900">
                                      {mode.label}
                                    </p>
                                    <span
                                      className={`h-3 w-3 rounded-full border ${
                                        selected
                                          ? "border-med-500 bg-med-500 shadow-[0_0_0_4px_rgba(45,212,191,0.18)]"
                                          : "border-slate-300 bg-white"
                                      }`}
                                    />
                                  </div>
                                  <p className="mt-1.5 text-sm leading-6 text-slate-600">
                                    {mode.description}
                                  </p>
                                </div>
                              </div>
                            </button>
                          )
                        })}
                      </div>
                    </div>
                  )}

                <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center">
                  {!isImageLoading &&
                    !classificationResult.final_class &&
                    !classificationResult.hasOwnProperty("detail") && (
                      <Button type="button" onClick={handleClassify}>
                        <Play size={16} />
                        {classifyButtonLabel}
                      </Button>
                    )}
                  {!isImageLoading && (
                    <Button type="button" variant="secondary" onClick={resetImage}>
                      <Trash2 size={16} />
                      Удалить
                    </Button>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {isImageLoading && (
        <div className="card-elevated flex flex-col items-center py-10 text-center">
          <Loader2 size={32} className="mb-3 animate-spin text-med-500" />
          <p className="font-medium text-gray-700">
            {analysisStage.title || "Изображение обрабатывается..."}
          </p>
          {analysisStage.description && (
            <p className="mt-1.5 max-w-lg text-sm text-gray-500">
              {analysisStage.description}
            </p>
          )}
          {activeJobLabel && (
            <p className="mt-1.5 max-w-md break-words text-sm text-gray-500">
              {activeJobLabel}
            </p>
          )}
          <div className="mt-5 flex flex-wrap justify-center gap-2">
            {analysisSteps.map((step) => {
              const isActive = step.key === analysisStage.key
              const isPassed =
                analysisSteps.findIndex((item) => item.key === analysisStage.key) >
                analysisSteps.findIndex((item) => item.key === step.key)
              return (
                <span
                  key={step.key}
                  className={`rounded-full px-3 py-1 text-xs font-medium ${
                    isActive
                      ? "bg-med-100 text-med-800 ring-1 ring-med-200"
                      : isPassed
                        ? "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200"
                        : "bg-gray-100 text-gray-500"
                  }`}
                >
                  {step.label}
                </span>
              )
            })}
          </div>
          <p className="mt-3 text-xs text-gray-400">
            Обычно это занимает до 2 минут
          </p>
        </div>
      )}

      {classificationResult.final_class && imageSrc && (
        <div className="card-elevated space-y-4" ref={resultsRef}>
          <div className="flex items-center gap-2">
            <CheckCircle2 size={20} className="text-green-500" />
            <h2 className="text-lg font-semibold text-gray-900">
              Результат классификации
            </h2>
          </div>

          <div className="flex flex-wrap gap-2">
            {getValues(classificationResult).map((val, i) => (
              <React.Fragment key={i}>
                {i > 0 && <span className="self-center text-gray-300">&rarr;</span>}
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
            descriptionState.importantLabels.length > 0 ||
            descriptionState.bucketedLabels.length > 0) && (
            <div className="space-y-3 rounded-xl border border-gray-200 bg-gray-50/70 p-4">
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
                  className="whitespace-pre-line text-sm leading-relaxed text-gray-700"
                />
              )}

              {shouldShowMissingDescriptionFallback && (
                <Alert variant="warning" title="Описание не сформировано">
                  Сервис вернул признаки, но текстовое описание не сформировал.
                  Попробуйте повторить анализ немного позже.
                </Alert>
              )}

              {descriptionState.importantLabels.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {descriptionState.importantLabels.map((label) => (
                    <span
                      key={label}
                      className="rounded-full border border-gray-200 bg-white px-3 py-1 text-xs font-medium text-gray-600"
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

export default ClassificationWorkspace
