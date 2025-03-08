import { BACKEND_URL } from "../imports/URLS"
import { UPLOAD_FILE } from "../imports/ENDPOINTS"

export const handleUploadImage = async ({id, fileData}) => {

    const formData = new FormData()
    formData.append("user_id", id)
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

      return responseJSON
    }
    catch (err) {
      alert(`Ошибка: ${err}`)
      return {result: null}
    }
  }