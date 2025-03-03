export const handleUploadImage = async (image) => {

    const formData = new FormData();
    formData.append("file", image);

    try {
      let response = await fetch(`http://localhost:9000/uploadfile`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        alert(`Произошла ошибка: ${response.status}`)
        return
      }

      let responseJSON = await response.json()

      console.log(responseJSON)

      return responseJSON
    }
    catch (err) {
      alert(`Ошибка: ${err}`)
      return {result: null}
    }
  }