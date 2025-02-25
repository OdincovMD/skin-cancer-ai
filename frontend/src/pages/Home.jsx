// pages/Home.jsx
import React from 'react'
import ReactImageMagnify from "easy-magnify-waft"

class Home extends React.Component {

  constructor(props) {
    super(props)

    this.state = {
      imageSrc: null,
      imageRef: null,
    }

    this.handleChange = this.handleChange.bind(this)
    this.handleUploadImage = this.handleUploadImage.bind(this)
  }

  async handleChange(event) {
    const fileInfo = event.target.files[0] 
    
    var now = new Date()
    const day = now.getDate() + '-' + (now.getMonth() + 1) + '-' + now.getFullYear()
    const time = now.getHours() + '-' + now.getMinutes() + '-' + now.getSeconds()
    const user = this.state.firstName + this.state.lastName
    // const filename = fileInfo.name.split(".")[0]
    // const ext = fileInfo.name.split(".").pop()
    const stamp = `${day}_${time}_${user}_${fileInfo.name}`
    const fileProcessed = new File([fileInfo], stamp, {type: fileInfo.type});

    this.setState({ formData: fileProcessed })
    
    if (fileProcessed && fileProcessed.type.startsWith('image/')) {
      const reader = new FileReader();

      reader.onload = (e) => {
        this.setState({ imageSrc: e.target.result},
        //  () => {console.log(this.state.imageSrc);}
        )
      }

      reader.readAsDataURL(fileInfo)
    }

  }

  async handleUploadImage() {

    const formData = new FormData();
    formData.append("file", this.state.formData);

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
    }
    catch (err) {
      alert(`Ошибка: ${err}`)
    }
  }

  render() {
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
      <div className="space-y-6 mt-5">
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex flex-col items-center justify-center">
            <h2 className="text-2xl font-bold text-gray-800 mb-4 text">
              Загрузите ваше изображение
            </h2>
            <div>
            {!this.state.imageSrc && (
              <form ref={(el) => this.myForm = el}>
                  <input type="file" onChange={(event) => { 
                    this.handleChange(event)
                  }} />
              </form>
            )}
            </div>
            <div id="small_image" className="w-1/2 h-auto max-w-[600px]">
            {this.state.imageSrc && (
                <ReactImageMagnify {...{
                      smallImage: {
                          alt: 'Загруженное изображение',
                          isFluidWidth: true,
                          src: this.state.imageSrc,
                          sizes: '(max-width: 480px) 100vw, (max-width: 1200px) 30vw, 360px'
                      },
                      largeImage: {
                          src: this.state.imageSrc,
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
            {this.state.imageSrc && (
              <div id="enlargened_image">
              </div>
            )}
            <div>
            {this.state.imageSrc && (
              <button onClick={() => {
                this.handleUploadImage()
                this.myForm.reset()
                }} className="mt-4 px-6 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors">
                Обработать изображение
              </button>
            )}
            </div>
            <div>
            {this.state.imageSrc && (
              <button onClick={() => {
                this.setState({imageSrc: null})
              }} className="mt-4 px-6 py-2 bg-red-500 text-white rounded-md hover:bg-red-600 transition-colors">
              Удалить изображение
              </button>
            )}
            </div>
          </div>
        </div>
      </div>
    </div>
    )
}
}

export default Home;