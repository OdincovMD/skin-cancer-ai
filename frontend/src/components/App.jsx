// components/App.jsx
import React, { useState } from 'react'
import { Routes, Route } from 'react-router-dom'
import Header from './Header.jsx'
import Sidebar from './Sidebar.jsx'

import Home from '../pages/Home'
import SignIn from '../pages/SignIn'
import SignUp from '../pages/SignUp'
import About from '../pages/About'

const App = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Header toggleSidebar={toggleSidebar} />
      <Sidebar isOpen={isSidebarOpen} />
      <main
        className={`pt-16 min-h-screen transition-all duration-300 
        ${isSidebarOpen ? 'pl-64' : 'pl-0'}`}
      >
        <div className="p-6">
          <Routes path="/" element={<Layout />}>
            <Route index element={<Home />}/>
            <Route path="/signin" element={<SignIn />} />
            <Route path="/signup" element={<SignUp />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}

const Layout = () => {
  return (
    <div>
      <nav>
        {/* Navigation links */}
      </nav>
      <Outlet /> {/* Where nested routes will render */}
    </div>
  );
 }

export default App