// components/Sidebar.jsx
import React from 'react'
import { Link } from 'react-router-dom'

const Sidebar = ({ isOpen }) => {
  const menuItems = [
    { text: 'Главная', path: '/' },
    { text: 'Войти', path: '/signin' },
    { text: 'Регистрация', path: '/signup' },
    { text: 'О нас', path: '/about' },
    // { text: 'Настройки', path: '/settings' },
  ]

  return (
    <aside
      className={`fixed left-0 top-16 h-[calc(100vh-4rem)] bg-white shadow-lg transition-transform duration-300 
      ${isOpen ? 'translate-x-0 w-64' : '-translate-x-full w-64'}`}
    >
      <div className="h-full overflow-hidden">
        <nav className="p-4">
          <ul className="space-y-2">
            {menuItems.map((item, index) => (
              <li key={index}>
                <Link
                  to={item.path}
                  className="block w-full p-3 rounded-lg hover:bg-gray-100 text-gray-700 transition-colors"
                >
                  <span className="block truncate">{item.text}</span>
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </aside>
  )
}

export default Sidebar