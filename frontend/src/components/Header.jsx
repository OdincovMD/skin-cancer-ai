// components/Header.jsx
import React from 'react'

const Header = ({ toggleSidebar }) => {
  return (
    <header className="fixed top-0 left-0 right-0 h-16 bg-white shadow-md z-50">
      <div className="flex items-center h-full px-4">
        <button
          onClick={toggleSidebar}
          className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
          aria-label="Toggle menu"
        >
          <span className="text-xl">☰</span>
        </button>
        <h1 className="ml-4 text-xl font-bold text-gray-800">Skin</h1>

        <div className="ml-auto flex items-center space-x-4">
          <span className="text-gray-600">Иван Иванов</span>
          <div className="w-8 h-8 rounded-full bg-gray-300"></div>
        </div>
      </div>
    </header>
  )
}

export default Header