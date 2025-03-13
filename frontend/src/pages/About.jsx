
import {React} from "react";
import {useDispatch, useSelector} from "react-redux"
import { Link } from "react-router-dom"

const About = () => {
  return (
    <div className="min-h-[88vh] flex flex-col items-center justify-center bg-gray-100 p-6">
      <div className="bg-white shadow-lg rounded-2xl p-8 max-w-2xl text-center">
        <h1 className="text-3xl font-bold text-gray-800 mb-4">О нас</h1>
        <p className="text-gray-600 text-lg mb-6">
          Мы — команда разработчиков, сделавшая этот сайт по распознаванию новообразований кожи.
        </p>
        {/* <h2 className="text-2xl font-semibold text-gray-700 mb-2">Разработчики</h2>
        <ul className="text-gray-600 text-lg">
        <li>Кегелик Николай Александрович</li> подумать что тут написать
          <li>Бикулич Глеб Игоревич</li>
          <li>Хасянов Булат Гаярович</li>
          <li>Чуканов Тимофей Витальевич</li>
        </ul>  */}
        <h2 className="text-2xl font-semibold text-gray-700 mt-6 mb-2">Университет</h2>
        <p className="text-gray-600 text-lg mb-6">НИЯУ МИФИ</p>

        <h3 className="italic text-gray-600 text-lg mb-8">
          С полным деревом классификации вы можете ознакомиться по
           <a 
              href="https://miro.com/app/board/uXjVMwEeFQ8=/"
              target="_blank"
              className="text-blue-600 underline ml-1 transition hover:text-blue-900"
            >
              ссылке
           </a>.
        </h3>

        <h4 className="italic text-gray-600">
          <a
            href="https://t.me/horokami"
            target="_blank"
            className="text-blue-600 underline ml-1 transition hover:text-blue-900"
          >
            Не работает?
          </a>
        </h4>
      </div>
    </div>
  );
}

export default About
