
import React from "react";

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
        <p className="text-gray-600 text-lg">НИЯУ МИФИ</p>
      </div>
    </div>
  );
}

export default About;
