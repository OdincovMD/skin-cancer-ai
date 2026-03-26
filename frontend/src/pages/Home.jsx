// pages/Home.jsx
import React, { useEffect, useRef, useState } from "react"
import ReactImageMagnify from "easy-magnify-waft"

import { fetchActiveClassificationJob } from "../asyncActions/fetchActiveClassificationJob"
import { handleUploadImage } from "../asyncActions/handleUploadImage"
import {
  clearPendingJob,
  pollClassificationJob,
  savePendingJob,
} from "../asyncActions/pollClassificationJob"
import { Link } from "react-router-dom"
import { useSelector } from "react-redux"
import TreeComponent from "../components/Tree"

import { env } from "../imports/ENV"
import { HISTORY_IMAGE, PROFILE } from "../imports/ENDPOINTS"
import { getValues } from "../imports/HELPERS"

function displayNameFromStoredFileName(storedName) {
  if (!storedName) return null
  const m = /^(?:.*?_){3}(?<filename>.*)$/.exec(storedName)
  return m?.groups?.filename ?? storedName
}

const Home = () => {

  const userInfo = useSelector(state => state.user)
  const defaultResult = {feature_type: null, structure: null, properties: [], final_class: null}
  const resumeEffectGen = useRef(0)

  const [isImageLoading, setIsImageLoading] = useState(false)
  const [activeJobLabel, setActiveJobLabel] = useState(null)
  const [fileName, setFileName] = useState(null)
  
  const [fileData, setFileData] = useState(null)
  const [imageSrc, setImageSrc] = useState(null)
  const [classificationResult, setClassificationResult] = useState(defaultResult)

  // useEffect(() => {
  //   console.log(fileData)
  // }, [fileData])

  // useEffect(() => {
  //   console.log(imageSrc)
  // }, [imageSrc])

  // useEffect(() => {
  //   console.log(classificationResult)
  // }, [classificationResult])

  // useEffect(() => {
  //   console.log(isImageLoading)
  // }, [isImageLoading])

  useEffect(() => {
    const uid = userInfo?.userData?.id
    const token = userInfo?.accessToken
    if (!uid || !token || !userInfo.emailVerified) return
    resumeEffectGen.current += 1
    const gen = resumeEffectGen.current
    let cancelled = false

    ;(async () => {
      try {
        const active = await fetchActiveClassificationJob(token)
        if (cancelled || resumeEffectGen.current !== gen) return
        if (!active) {
          clearPendingJob(uid)
          return
        }
        savePendingJob(uid, active.job_id)
        setActiveJobLabel(displayNameFromStoredFileName(active.file_name))
        setIsImageLoading(true)
        const polled = await pollClassificationJob({
          jobId: active.job_id,
          userId: uid,
          accessToken: token,
        })
        if (cancelled || resumeEffectGen.current !== gen) return
        setClassificationResult(polled.classification)
        if (polled.imageToken) {
          const b = env.BACKEND_URL.replace(/\/$/, "")
          setImageSrc(
            `${b}${HISTORY_IMAGE}?token=${encodeURIComponent(polled.imageToken)}`
          )
        }
      } catch (e) {
        if (!cancelled && resumeEffectGen.current === gen) {
          alert(String(e.message || e))
          setClassificationResult(defaultResult)
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

  const handleChange = async (event) => {

    // Это для чтения информации о файле и подготовки файла к отправке на бэк
    const fileInfo = event.target.files[0]
    setFileName(fileInfo.name)

    var now = new Date()
    const day = now.getDate() + "-" + (now.getMonth() + 1) + "-" + now.getFullYear()
    const time = now.getHours() + "-" + now.getMinutes() + "-" + now.getSeconds()
    const user = userInfo.userData.id
    const stamp = `${day}_${time}_${user}_${fileInfo.name}`
    const fileProcessed = new File([fileInfo], stamp, {type: fileInfo.type});
    setFileData(fileProcessed)
    
    // Это для отображения изображения на сайте
    if (fileProcessed && fileProcessed.type.startsWith("image/")) {
      const reader = new FileReader();

      reader.onload = (e) => {
        setImageSrc(e.target.result)
      }

      reader.readAsDataURL(fileProcessed)
    }

    setClassificationResult(defaultResult)
  }

  const info = 
  <div className="space-y-6">
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">
        Диагностика новообразований кожи
      </h2>
      <div className="grid md:grid-cols-2 gap-6">
        <div className="space-y-4">
          <p className="text-gray-600">
            Наш сервис помогает в ранней диагностике новообразований кожи с использованием
            современных технологий машинного обучения.
          </p>
          <div className="space-y-2">
            <h3 className="font-semibold text-gray-700">Преимущества ранней диагностики:</h3>
            <ul className="list-disc list-inside text-gray-600 space-y-1">
              <li>Раннее выявление потенциальных проблем</li>
              <li>Быстрая предварительная оценка</li>
              <li>Возможность своевременного обращения к специалисту</li>
              <li>Регулярный мониторинг изменений</li>
            </ul>
          </div>
          <p className="mb-auto text-gray-600">
            Наши классификаторы обучались на изображениях размером 2560х1920.
            Разрешение изображения, отличное от указанного, <span className="italic">может</span> повлиять на результат распознавания.            
          </p>
          <p className="mb-auto text-gray-600">
            Видимое искажение изображения <span className="italic">не влияет</span> на результаты классификации.       
          </p>
        </div>
        <div className="relative h-64 md:h-auto">
          <img
            src="/images/example.jpg"
            alt="Пример новообразования"
            className="rounded-lg object-cover w-full h-full"
          />
          <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white p-2 rounded-b-lg">
            <p className="text-sm">
              Пример новообразования
            </p>
          </div>
        </div>
      </div>
      <div className="mt-6 p-4 bg-yellow-50 rounded-lg border border-yellow-100">
        <p className="text-sm text-yellow-800">
          <strong>Важно:</strong> Данный сервис не является заменой
          профессиональной медицинской консультации. При наличии любых сомнений
          обязательно обратитесь к квалифицированному дерматологу.
        </p>
      </div>
    </div>
  </div>

  const uploadBlocked =
    userInfo.userData.id &&
    userInfo.accessToken &&
    !userInfo.emailVerified && (
      <div className="space-y-6 mt-5">
        <div className="flex flex-col items-center justify-center rounded-lg border border-amber-200 bg-amber-50 p-6 shadow-md">
          <h2 className="mb-2 text-xl font-bold text-amber-900">
            Подтвердите email
          </h2>
          <p className="mb-4 max-w-lg text-center text-amber-800">
            Загрузка изображений в классификатор доступна только после подтверждения адреса
            электронной почты. Проверьте входящие и папку «Спам» или запросите письмо повторно в
            личном кабинете.
          </p>
          <Link
            to={PROFILE}
            className="rounded-lg bg-amber-700 px-5 py-2 font-semibold text-white transition hover:bg-amber-800"
          >
            Перейти в личный кабинет
          </Link>
        </div>
      </div>
    )

  const uploadImage =
    userInfo.userData.id &&
    userInfo.accessToken &&
    userInfo.emailVerified && (
    <div className="space-y-6 mt-5">
      <div className="flex flex-col items-center justify-center bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 text">
          Загрузите ваше изображение
        </h2>
        {!imageSrc &&
          <div>
            <form className="mt-2">
                <label
                  className="relative inline-block"
                >
                  <input
                  type="file"
                  className="absolute -z-10 opacity-0 block w-0 h-0"
                  onChange={(event) => { 
                    handleChange(event)
                  }} />
                  <span className="cursor-pointer rounded-lg px-4 py-3 text-white font-semibold transition bg-blue-500 hover:bg-blue-600">
                    Выберите файл
                  </span>
                </label>
            </form>
          </div>
        }
        {imageSrc &&
        <div className="flex flex-col items-center justify-center">
            <p className="font-semibold text-gray-700">
              {`Загруженный файл: ${fileName}`}
            </p>
            <div className="flex flex-row items-center justify-center mt-5">
              <div id="small_image" className="w-1/2 h-auto">
                  <ReactImageMagnify {...{
                  smallImage: {
                      alt: "Загруженное изображение",
                      isFluidWidth: true,
                      src: imageSrc,
                  },
                  largeImage: {
                      src: imageSrc,
                      width: 2560,
                      height: 1920
                  },
                  enlargedImagePortalId: "enlargened_image",
                  isHintEnabled: true,
                  shouldHideHintAfterFirstActivation: false,
                  isActivatedOnTouch: true,
                  }}/>
              </div>
              <div 
                id="enlargened_image"
              >
              </div>
          </div>
        </div>
        }
        {classificationResult.hasOwnProperty("detail") &&
          <p className="text-red-600">
            {classificationResult.detail}
          </p>
        }
        {(imageSrc && !isImageLoading && !classificationResult.final_class && !classificationResult.hasOwnProperty("detail")) &&
          <div>
            <button 
              onClick={() => {
                setIsImageLoading(true)
                handleUploadImage({
                  id: userInfo.userData.id,
                  fileData: fileData,
                  accessToken: userInfo.accessToken,
                })
                  .then((response) => {
                    if (
                      response &&
                      typeof response === "object" &&
                      Object.prototype.hasOwnProperty.call(
                        response,
                        "classification"
                      )
                    ) {
                      setClassificationResult(response.classification)
                      if (response.imageToken) {
                        const b = env.BACKEND_URL.replace(/\/$/, "")
                        setImageSrc(
                          `${b}${HISTORY_IMAGE}?token=${encodeURIComponent(
                            response.imageToken
                          )}`
                        )
                      }
                    } else {
                      setClassificationResult(response)
                    }
                    setIsImageLoading(false)
                  })
              }}
              className="mt-4 px-6 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
            >
              Обработать изображение
            </button>
          </div>
        }
        {imageSrc && !isImageLoading &&
          <div>
            <button 
              onClick={() => {
                setClassificationResult(defaultResult)
                setFileName(null)
                setImageSrc(null)
                setFileData(null)
              }} 
              className="mt-4 px-6 py-2 bg-red-500 text-white rounded-md hover:bg-red-600 transition-colors">
              Удалить изображение
            </button>
          </div>
        }
        </div>
    </div>
    )

  const result = classificationResult.final_class && imageSrc ?
    <div className="space-y-6 mt-5">
      <div className="flex flex-col items-center justify-center bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 text">
          Результат классификации
        </h2>
        <p className="font-semibold text-gray-700">
          {getValues(classificationResult).reduce((accumulator, currentValue) => (accumulator + " -> " + currentValue))}
        </p>
        <TreeComponent classificationResult={classificationResult} displaySize={{width: "100%", height: "500px"}} nodeSize={{x: 300, y: 50}} zoom={0.6} translate={{x: 300, y: 300}}/>
      </div>
    </div> : isImageLoading ?
    <div className="space-y-6 mt-5">
      <div className="flex flex-col items-center justify-center bg-white rounded-lg shadow-md p-6">
        <img
          src="/images/loading.gif"
          alt="Процесс загрузки"
          className="rounded-lg object-cover w-8 h-8"
        />
        <div className="font-semibold text-gray-700">
          Подождите, ваше изображение обрабатывается...
        </div>
        {activeJobLabel &&
          <p className="mt-2 text-sm text-gray-600 text-center max-w-md break-words">
            Файл: {activeJobLabel}
          </p>
        }
      </div>
    </div> : null

  return (
    <div>
    {info}
    {uploadBlocked}
    {uploadImage}
    {result}
    </div>
  )
}

export default Home