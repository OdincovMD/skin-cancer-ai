import React, { useEffect, useState } from "react"
import { useDispatch, useSelector } from "react-redux"
import { handleHistoryRequest } from "../asyncActions/handleHistoryRequest"

import { getValues, mappingInfoRU } from "../imports/HELPERS"

import TreeComponent from "../components/Tree"

const Profile = () => {
  
  const dispatch = useDispatch()
  const userInfo = useSelector(state => state.user)

  // request_date, file_name, status, result
  const [history, setHistory] = useState([])

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
  
  const showHistory = (historyResponse) => {

    const data_time = new RegExp("^(?<data>.*)T(?<time>.*)\\..*\\+(?<correction>.*)$")
    const file_name = new RegExp("^(?:.*?_){3}(?<filename>.*)$")

    const requestDate = data_time.exec(historyResponse.request_date)
    const fileName = file_name.exec(historyResponse.file_name)
    const result = requestDate ? JSON.parse(historyResponse.result) : {}

    return ( 
      <ul className="flex flex-row justify-between items-center rounded-lg border border-gray-900 p-3">
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
          (historyResponse.status == "error") &&
          <p className="text-red-600">
            Произошла ошибка со стороны бэкенда. Свяжитесь с администрацией сайта.
          </p>
          }
          {
          result.hasOwnProperty("detail") && 
          <p className="text-red-600">
            Произошла ошибка обработки фотографии. Свяжитесь с администрацией сайта.
          </p>
          }
          {
          !result.hasOwnProperty("detail") && (Object.keys(result).length > 0) &&
          <div className="w-[100%] font-semibold text-gray-700">
            {getValues(result).reduce((accumulator, currentValue) => (accumulator + " ->\n" + currentValue))}
          </div>
          }
          {
          <div className="flex flex-col justify-center items-center w-[100%]">
            {
              !result.hasOwnProperty("detail") && ((Object.keys(result).length > 0) ?
              <TreeComponent classificationResult={result} displaySize={{width: "100%", height: "300px"}} nodeSize={{x: 300, y: 50}} zoom={0.4} translate={{x: 50, y: 180}}/> :
              mappingInfoRU.result)
            }
          </div>
          }
        </li>
      </ul>
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
            <li key={index}>
              {showHistory(historyResponse)}
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
                handleHistoryRequest(userInfo.userData.id).then((response) => setHistory([
                  {
                    request_date: mappingInfoRU.request_date,
                    file_name: mappingInfoRU.file_name,
                    status: mappingInfoRU.status,
                    result: mappingInfoRU.result,
                  },
                  ...response
                  ]))
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