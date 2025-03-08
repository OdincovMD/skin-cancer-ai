import { BACKEND_URL } from "../imports/URLS"
import { UPLOAD_FILE } from "../imports/ENDPOINTS"

export const handleUploadImage = async (fileData) => {

    const formData = new FormData()
    formData.append("file", fileData)

    try {
      let response = await fetch(`${BACKEND_URL}${UPLOAD_FILE}`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        alert(`Произошла ошибка: ${response.status}`)
        return
      }

      let responseJSON = await response.json()

      // console.log(responseJSON)

      return responseJSON
    }
    catch (err) {
      alert(`Ошибка: ${err}`)
      return {result: null}
    }
  }