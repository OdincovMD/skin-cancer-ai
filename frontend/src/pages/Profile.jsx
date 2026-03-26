import React, { useState } from "react"
import { useSelector } from "react-redux"
import { handleHistoryRequest } from "../asyncActions/handleHistoryRequest"

import { env } from "../imports/ENV"
import { HISTORY_IMAGE } from "../imports/ENDPOINTS"
import { getValues, mappingInfoRU } from "../imports/HELPERS"

import TreeComponent from "../components/Tree"

const Profile = () => {
  
  const userInfo = useSelector(state => state.user)

  // request_date, file_name, bucket_name?, status, result
  const [history, setHistory] = useState([])
  const [openHistoryImageKey, setOpenHistoryImageKey] = useState(null)

  // useEffect(() => {
  //   console.log(history)
  // }, [history])

  const showInfo = (field) => {
    return (
      mappingInfoRU[field] ?
        <div className="flex flex-row justify-start items-center gap-[10px]">
          <div className="rounded-lg border-none w-[170px] p-3">
            <span className="block truncate font-semibold">{mappingInfoRU[field]}</span>
          </div>
          <div className="rounded-lg border flex-grow border-gray-300 p-3">
            <span className="block truncate">{userInfo.userData[field]}</span>
          </div>
        </div> :
        null
    )
  }
  
  const showHistory = (historyResponse, userId) => {
    const isHeaderRow = historyResponse.file_name === mappingInfoRU.file_name
    const rowKey = `${String(historyResponse.request_date)}_${historyResponse.file_name}`
    const base = env.BACKEND_URL.replace(/\/$/, "")
    const imageSrc =
      userId &&
      historyResponse.file_name &&
      !isHeaderRow &&
      historyResponse.image_token
        ? `${base}${HISTORY_IMAGE}?token=${encodeURIComponent(
            historyResponse.image_token
          )}`
        : null

    const data_time = new RegExp("^(?<data>.*)T(?<time>.*)\\..*\\+(?<correction>.*)$")
    const file_name = new RegExp("^(?:.*?_){3}(?<filename>.*)$")

    const requestDate = data_time.exec(historyResponse.request_date)
    const fileName = file_name.exec(historyResponse.file_name)

    const parseClassificationResult = () => {
      const raw = historyResponse.result
      if (raw == null || raw === "") return {}
      try {
        const parsed = typeof raw === "string" ? JSON.parse(raw) : raw
        if (parsed != null && typeof parsed === "object" && !Array.isArray(parsed)) {
          return parsed
        }
        return {}
      } catch {
        return {}
      }
    }

    const result = parseClassificationResult()
    const status = historyResponse.status
    const inProgress =
      status === "pending" || status === "processing"

    return (
      <div className="rounded-lg border border-gray-900 p-3 space-y-3">
      <ul className="flex flex-row justify-between items-center">
        <li 
          key={0}
          className="w-[15%] text-center"
        >
          {
            requestDate ? 
            `${requestDate.groups.data}, ${requestDate.groups.time}` :
            mappingInfoRU.request_date
          }
        </li>
        <li 
          key={1}
          className="w-[10%] text-center"
        >
          {
            fileName ? 
            `${fileName.groups.filename}` :
            mappingInfoRU.file_name
          }
        </li>
        <li 
          key={3}
          className="flex flex-col justify-center items-center gap-[10px] w-[55%] text-center"
        > 
          {
          (status === "error") &&
          <p className="text-red-600">
            Произошла ошибка со стороны бэкенда. Свяжитесь с администрацией сайта.
          </p>
          }
          {
          inProgress &&
          <p className="text-amber-700">
            Классификация выполняется или ожидает обработки. Обновите историю позже.
          </p>
          }
          {
          (status !== "error" && !inProgress) && Object.prototype.hasOwnProperty.call(result, "detail") &&
          <p className="text-red-600">
            Произошла ошибка обработки фотографии. Свяжитесь с администрацией сайта.
          </p>
          }
          {
          (status !== "error" && !inProgress) && !Object.prototype.hasOwnProperty.call(result, "detail") && (Object.keys(result).length > 0) &&
          <div className="w-[100%] font-semibold text-gray-700">
            {getValues(result).reduce((accumulator, currentValue) => (accumulator + " ->\n" + currentValue))}
          </div>
          }
          {
          (status !== "error" && !inProgress) &&
          <div className="flex flex-col justify-center items-center w-[100%]">
            {
              !Object.prototype.hasOwnProperty.call(result, "detail") ? (
                Object.keys(result).length > 0 ? (
                  <TreeComponent classificationResult={result} displaySize={{width: "100%", height: "300px"}} nodeSize={{x: 300, y: 50}} zoom={0.4} translate={{x: 50, y: 180}}/>
                ) : (
                  mappingInfoRU.result
                )
              ) : null
            }
          </div>
          }
        </li>
      </ul>
      {!isHeaderRow && imageSrc && (
        <div className="flex flex-col items-center gap-2 border-t border-gray-200 pt-3">
          <button
            type="button"
            onClick={() =>
              setOpenHistoryImageKey((k) => (k === rowKey ? null : rowKey))
            }
            className="px-4 py-2 text-sm bg-slate-600 text-white rounded-md hover:bg-slate-700"
          >
            {openHistoryImageKey === rowKey
              ? "Скрыть изображение"
              : "Показать изображение из хранилища"}
          </button>
          {openHistoryImageKey === rowKey && (
            <img
              src={imageSrc}
              alt={historyResponse.file_name || "Снимок"}
              className="max-h-[min(480px,70vh)] w-auto max-w-full rounded border border-gray-300 object-contain"
              onError={() => {
                alert("Не удалось загрузить изображение (файл отсутствует в MinIO или ошибка сети).")
                setOpenHistoryImageKey(null)
              }}
            />
          )}
        </div>
      )}
      </div>
    )
  }

  const profilePicture = 
    <div className="space-y-6 flex-shrink-0">
      <div className="bg-white rounded-lg shadow-md p-6 h-[220px]">
        <img
          src={userInfo.userData.id == 1 ? "/images/PP.png" : "/images/image.png"}
          alt="Фотография профиля"
          className="rounded-lg object-cover w-full h-full border border-gray-700"
        />
      </div>
    </div>

  const profileInfo =
    <div className="space-y-6">
      <div className="flex flex-row items-center gap-[10px] bg-white rounded-lg shadow-md p-6">
        <ul className="space-y-2 flex-grow">
          {Object.keys(userInfo.userData).map((field, index) => (
            <li key={index} >
              {showInfo(field)}
            </li>
          ))}
        </ul>
      </div>
    </div>

  const historyDisplay = history.length > 0 ?
    <div className="space-y-6 mt-5">
      <div className="flex flex-column items-center justify-center bg-white rounded-lg shadow-md p-6">
        <ul className="w-full space-y-2">
          {history.map((historyResponse, index) => (
            <li key={`${String(historyResponse.request_date)}_${historyResponse.file_name}_${index}`}>
              {showHistory(historyResponse, userInfo.userData.id)}
            </li>
          ))}
        </ul>
      </div>
    </div> :
    null

  const requestButton = 
    <div className="space-y-6 mt-5 w-full">
      <div className="flex flex-column items-center justify-center bg-white rounded-lg shadow-md p-6">
        <div>
            <button 
              onClick={() => {
                handleHistoryRequest(userInfo.userData.id).then((response) => {
                  const rows = Array.isArray(response) ? response : []
                  setOpenHistoryImageKey(null)
                  setHistory([
                    {
                      request_date: mappingInfoRU.request_date,
                      file_name: mappingInfoRU.file_name,
                      status: mappingInfoRU.status,
                      result: mappingInfoRU.result,
                    },
                    ...rows,
                  ])
                })
              }} 
              className="px-6 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
            >
              Получить историю запросов
            </button>
        </div>
      </div>
    </div> 

  return (
    <div className="flex flex-col justify-center items-center">
      <div className="flex flex-row justify-center gap-[20px] w-[60%]">
          {profilePicture}
          {profileInfo}
      </div>
      <div className="flex flex-col justify-center w-[80%]">
        {historyDisplay}
        {requestButton}
      </div>
    </div>
  )
}

export default Profile