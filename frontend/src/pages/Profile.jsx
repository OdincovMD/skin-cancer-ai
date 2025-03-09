import React, { useEffect, useState } from "react"
import { useDispatch, useSelector } from "react-redux"
import { handleHistoryRequest } from "../asyncActions/handleHistoryRequest"

const Profile = () => {
  
  const dispatch = useDispatch()
  const userInfo = useSelector(state => state.user)

  // request_date, file_name, status, result
  const defaultResult = [null, null, null, null]

  const [history, setHistory] = useState([defaultResult, defaultResult, defaultResult])

  const requestButton = 
  <div className="space-y-6">
  <div className="bg-white rounded-lg shadow-md p-6">
    <div>
        <button 
          onClick={() => {
            handleHistoryRequest(userInfo.userData.id).then((response) => setHistory(response))
          }} 
          className="mt-4 px-6 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
        >
          Получить историю запросов
        </button>
    </div>
  </div>
  </div>

  return (
    <div>
    {requestButton}
    </div>
  )
}

export default Profile