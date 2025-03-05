// pages/Home.jsx
import React, { useState } from 'react'
import ReactImageMagnify from "easy-magnify-waft"
import { handleUploadImage } from '../asyncActions/handleUploadImage'
import { useSelector } from 'react-redux'

const Home = () => {

  const userInfo = useSelector(state => state.user)
  
  const [imageSrc, setImageSrc] = useState(null)
  const [formData, setFormData] = useState(null)
  var myForm = null

  const handleChange = (event) => {
    const fileInfo = event.target.files[0] 
    
    var now = new Date()
    const day = now.getDate() + '-' + (now.getMonth() + 1) + '-' + now.getFullYear()
    const time = now.getHours() + '-' + now.getMinutes() + '-' + now.getSeconds()
    const user = userInfo.name + "-" + userInfo.surname
    // const filename = fileInfo.name.split(".")[0]
    // const ext = fileInfo.name.split(".").pop()
    const stamp = `${day}_${time}_${user}_${fileInfo.name}`
    const fileProcessed = new File([fileInfo], stamp, {type: fileInfo.type});

    setFormData(fileProcessed)
    
    if (fileProcessed && fileProcessed.type.startsWith('image/')) {
      const reader = new FileReader();

      reader.onload = (e) => {
        setImageSrc(e)
      }

      reader.readAsDataURL(fileInfo)
    }

  }

  const classification = userInfo.name 
    ?<div className="space-y-6 mt-5">
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex flex-col items-center justify-center">
          <h2 className="text-2xl font-bold text-gray-800 mb-4 text">
            Загрузите ваше изображение
          </h2>
          <div>
            {!imageSrc && (
              <form ref={(el) => {myForm = el}}>
                  <input type="file" onChange={(event) => { 
                    handleChange(event)
                  }} />
              </form>
            )}
          </div>
          <div className="flex flex-row items-center justify-center">
            <div id="small_image" className="w-1/2 h-auto">
              {imageSrc && (
                <ReactImageMagnify {...{
                smallImage: {
                    alt: 'Загруженное изображение',
                    isFluidWidth: true,
                    src: imageSrc,
                },
                largeImage: {
                    src: imageSrc,
                    width: 2560,
                    height: 1920
                },
                enlargedImagePortalId: 'enlargened_image',
                isHintEnabled: true,
                shouldHideHintAfterFirstActivation: false,
                isActivatedOnTouch: true,
                }}/>
              )}
            </div>
              {imageSrc && (
              <div id="enlargened_image">
              </div>
              )}
          </div>
          <div>
            {imageSrc && (
              <button onClick={() => {
                handleUploadImage(formData)
                myForm.reset()
                }} className="mt-4 px-6 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors">
                Обработать изображение
              </button>
            )}
          </div>
          <div>
          {imageSrc && (
            <button onClick={() => {
              setImageSrc(null)
            }} className="mt-4 px-6 py-2 bg-red-500 text-white rounded-md hover:bg-red-600 transition-colors">
            Удалить изображение
            </button>
          )}
          </div>
        </div>
      </div>
    </div>
    : null

  return (
    <div>
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
    {classification}
  </div>
  )
}

export default Home;