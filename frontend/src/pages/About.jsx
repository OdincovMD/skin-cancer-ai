import React from "react"
import { GraduationCap, ExternalLink, MessageCircle, GitBranch } from "lucide-react"

import Button from "../components/ui/Button"

const About = () => {
  return (
    <div className="flex min-h-[calc(100vh-3.5rem)] items-center justify-center px-4 py-12">
      <div className="w-full max-w-lg">
        <div className="card-elevated text-center space-y-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">О проекте</h1>
            <p className="mt-2 text-gray-600 leading-relaxed">
              Мы — команда разработчиков, создавшая сервис автоматической
              диагностики новообразований кожи с использованием метода Киттлера
              и современных алгоритмов машинного обучения.
            </p>
          </div>

          <div className="flex items-center justify-center gap-2 text-gray-700">
            <GraduationCap size={20} className="text-med-600" />
            <span className="font-semibold">НИЯУ МИФИ</span>
          </div>

          <div className="space-y-3">
            <Button
              variant="secondary"
              href="https://miro.com/app/board/uXjVMwEeFQ8=/"
              external
              className="w-full !justify-between"
            >
              <span className="flex items-center gap-2">
                <GitBranch size={16} />
                Полное дерево классификации
              </span>
              <ExternalLink size={14} className="text-gray-400" />
            </Button>

            <Button
              variant="secondary"
              href="https://t.me/horokami"
              external
              className="w-full !justify-between"
            >
              <span className="flex items-center gap-2">
                <MessageCircle size={16} />
                Сообщить о проблеме
              </span>
              <ExternalLink size={14} className="text-gray-400" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default About
