import React, { useCallback, useMemo, useState } from "react"
import { Link } from "react-router-dom"
import {
  ArrowLeft,
  BookOpen,
  Check,
  Clock,
  Copy,
  Key,
  Rocket,
  Send,
  ShieldCheck,
  Zap,
} from "lucide-react"

import { env } from "../imports/ENV"
import { API_V1_PREFIX, HOME } from "../imports/ENDPOINTS"

const CodeBlock = ({ children, label }) => {
  const [copied, setCopied] = useState(false)

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(children).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }, [children])

  return (
    <div className="group relative rounded-lg bg-gray-900 text-gray-100">
      {label && (
        <div className="flex items-center justify-between border-b border-gray-700/60 px-4 py-2">
          <span className="text-xs font-medium text-gray-400">{label}</span>
          <CopyBtn copied={copied} onClick={handleCopy} />
        </div>
      )}
      <pre className="overflow-x-auto p-4 text-[13px] leading-relaxed">
        {children}
      </pre>
      {!label && (
        <div className="absolute right-2 top-2 opacity-0 transition-opacity group-hover:opacity-100">
          <CopyBtn copied={copied} onClick={handleCopy} />
        </div>
      )}
    </div>
  )
}

const CopyBtn = ({ copied, onClick }) => (
  <button
    onClick={onClick}
    className={`inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs transition-colors ${
      copied
        ? "bg-green-600/20 text-green-400"
        : "bg-gray-700/60 text-gray-400 hover:bg-gray-700 hover:text-gray-200"
    }`}
  >
    {copied ? <Check size={13} /> : <Copy size={13} />}
    {copied ? "Скопировано" : "Копировать"}
  </button>
)

const InlineCode = ({ children }) => (
  <code className="rounded bg-gray-100 px-1.5 py-0.5 text-[13px] font-medium text-gray-800">
    {children}
  </code>
)

const Badge = ({ children, color = "gray" }) => {
  const colors = {
    green: "bg-green-100 text-green-700",
    blue: "bg-med-50 text-med-700",
    yellow: "bg-amber-50 text-amber-700",
    red: "bg-red-50 text-red-700",
    gray: "bg-gray-100 text-gray-600",
  }
  return (
    <span
      className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-semibold ${colors[color]}`}
    >
      {children}
    </span>
  )
}

const MethodBadge = ({ method }) => {
  const map = { GET: "green", POST: "blue", DELETE: "red" }
  return <Badge color={map[method] || "gray"}>{method}</Badge>
}

const Section = ({ id, icon: Icon, title, children }) => (
  <section id={id} className="card-elevated space-y-5">
    <h2 className="flex items-center gap-2.5 text-lg font-semibold text-gray-900">
      {Icon && <Icon size={20} className="text-med-600" />}
      {title}
    </h2>
    {children}
  </section>
)

const P = ({ children }) => (
  <p className="text-sm leading-relaxed text-gray-600">{children}</p>
)

const ApiDocs = () => {
  const base = useMemo(() => {
    const origin =
      typeof window !== "undefined" ? window.location.origin : ""
    const api = env.BACKEND_URL.replace(/\/$/, "")
    return `${origin}${api}`
  }, [])

  const v1 = `${base}${API_V1_PREFIX}`

  return (
    <div className="mx-auto max-w-3xl space-y-8 pb-16">
      <header>
        <Link
          to={HOME}
          className="mb-6 inline-flex items-center gap-1.5 text-sm text-med-600 hover:text-med-700"
        >
          <ArrowLeft size={16} />
          На главную
        </Link>

        <div className="flex items-start gap-4">
          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-med-600 text-white">
            <BookOpen size={24} />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              API для разработчиков
            </h1>
            <p className="mt-1 text-sm text-gray-500">
              Классифицируйте дерматоскопические изображения из вашего кода за
              три шага: получите ключ, отправьте снимок, заберите результат.
            </p>
          </div>
        </div>
      </header>

      {/* ---- quickstart ---- */}
      <Section id="quickstart" icon={Rocket} title="Быстрый старт">
        <div className="space-y-6">
          <div className="flex gap-3">
            <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-med-100 text-xs font-bold text-med-700">
              1
            </span>
            <div className="space-y-2">
              <p className="text-sm font-medium text-gray-800">
                Получите ключ
              </p>
              <P>
                Подтвердите email и выпустите API-ключ в{" "}
                <Link to="/profile" className="text-link">
                  личном кабинете
                </Link>
                . Ключ начинается с <InlineCode>scai_</InlineCode> и
                показывается только один раз — сохраните его сразу.
              </P>
            </div>
          </div>

          <div className="flex gap-3">
            <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-med-100 text-xs font-bold text-med-700">
              2
            </span>
            <div className="min-w-0 flex-1 space-y-2">
              <p className="text-sm font-medium text-gray-800">
                Отправьте изображение
              </p>
              <CodeBlock label="bash">
                {`curl -X POST "${v1}/uploadfile" \\\n  -H "X-API-Key: scai_ваш_ключ" \\\n  -F "file=@photo.jpg"`}
              </CodeBlock>
              <CodeBlock label="ответ">{`{ "job_id": 42, "status": "pending" }`}</CodeBlock>
            </div>
          </div>

          <div className="flex gap-3">
            <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-med-100 text-xs font-bold text-med-700">
              3
            </span>
            <div className="min-w-0 flex-1 space-y-2">
              <p className="text-sm font-medium text-gray-800">
                Получите результат
              </p>
              <P>
                Опрашивайте задание, пока{" "}
                <InlineCode>status</InlineCode> не станет{" "}
                <InlineCode>completed</InlineCode>:
              </P>
              <CodeBlock label="bash">
                {`curl "${v1}/classification-jobs/42" \\\n  -H "X-API-Key: scai_ваш_ключ"`}
              </CodeBlock>
              <CodeBlock label="ответ">
                {`{
  "status": "completed",
  "result": { "final_class": "Melanocytic nevus", "..." : "..." },
  "image_token": "eyJ...",
  "description_status": "generating",
  "description": null,
  "description_error": null,
  "important_labels": []
}`}
              </CodeBlock>
            </div>
          </div>
        </div>
      </Section>

      {/* ---- auth ---- */}
      <Section id="auth" icon={Key} title="Аутентификация">
        <P>Добавляйте заголовок к каждому запросу:</P>
        <CodeBlock>{"X-API-Key: scai_ваш_ключ"}</CodeBlock>
        <P>
          Не оборачивайте его в <InlineCode>Bearer</InlineCode>, не ставьте
          кавычки. Просто строка целиком.
        </P>

        <div className="overflow-x-auto rounded-lg border border-gray-200">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 bg-gray-50">
                <th className="px-4 py-2.5 text-left font-semibold text-gray-700">
                  Код
                </th>
                <th className="px-4 py-2.5 text-left font-semibold text-gray-700">
                  Что произошло
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              <tr>
                <td className="px-4 py-2.5 font-mono text-red-600">401</td>
                <td className="px-4 py-2.5 text-gray-600">
                  Ключ не передан, неверный или отозван
                </td>
              </tr>
              <tr>
                <td className="px-4 py-2.5 font-mono text-red-600">403</td>
                <td className="px-4 py-2.5 text-gray-600">
                  Email владельца не подтверждён
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </Section>

      {/* ---- base url ---- */}
      <Section id="base-url" icon={Zap} title="Базовый URL">
        <P>
          Все методы начинаются с <InlineCode>/api/v1</InlineCode> после базы
          инсталляции:
        </P>
        <CodeBlock>{v1}</CodeBlock>
        <P>
          Если вы разворачиваете свой экземпляр, адрес будет другим — уточняйте
          у администратора.
        </P>
      </Section>

      {/* ---- endpoints ---- */}
      <Section id="endpoints" icon={Send} title="Методы API">
        <div className="space-y-8">
          <Endpoint
            method="POST"
            path="/api/v1/uploadfile"
            description="Загрузка изображения"
            v1={v1}
          >
            <P>
              Отправьте файл как <InlineCode>multipart/form-data</InlineCode> с
              полем <InlineCode>file</InlineCode>.
            </P>
            <CodeBlock label="bash">
              {`curl -X POST "${v1}/uploadfile" \\\n  -H "X-API-Key: scai_ваш_ключ" \\\n  -F "file=@image.jpg"`}
            </CodeBlock>
            <CodeBlock label="ответ">{`{ "job_id": 42, "status": "pending" }`}</CodeBlock>
            <Callout>
              Одновременно может быть только одно активное задание. Если
              предыдущее ещё не завершено — <InlineCode>429</InlineCode>.
            </Callout>
          </Endpoint>

          <Endpoint
            method="GET"
            path="/api/v1/classification-jobs/{job_id}"
            description="Статус задания"
            v1={v1}
          >
            <P>
              Подставьте числовой <InlineCode>job_id</InlineCode> из ответа
              загрузки. Рекомендуемый интервал опроса — <strong>2 секунды</strong>.
            </P>
            <CodeBlock label="ответ — готово">
              {`{
  "status": "completed",
  "result": {
    "feature_type": "...",
    "structure": "...",
    "final_class": "Melanocytic nevus"
  },
  "image_token": "eyJ...",
  "description_status": "completed",
  "description": "Клиническое описание...",
  "description_error": null,
  "important_labels": ["shape:неправильная"]
}`}
            </CodeBlock>
            <CodeBlock label="ответ — ошибка модели">
              {`{
  "status": "error",
  "result": { "detail": "Описание проблемы" }
}`}
            </CodeBlock>
          </Endpoint>

          <Endpoint
            method="GET"
            path="/api/v1/classification-jobs/active"
            description="Активное задание"
            v1={v1}
          >
            <P>
              Если есть незавершённое задание — вернёт его объект. Если нет —{" "}
              <InlineCode>204</InlineCode> с пустым телом. Удобно для
              восстановления контекста после перезапуска скрипта. Задание
              остаётся активным, пока не завершится и описание.
            </P>
          </Endpoint>

          <Endpoint
            method="POST"
            path="/api/v1/gethistory"
            description="История классификаций"
            v1={v1}
          >
            <P>
              Тело запроса: <InlineCode>{"{}"}</InlineCode> (пустой JSON).
              Возвращает массив прошлых классификаций. У каждой записи будет{" "}
              <InlineCode>image_token</InlineCode> для загрузки превью, а также
              поля <InlineCode>description</InlineCode>,{" "}
              <InlineCode>description_status</InlineCode>,{" "}
              <InlineCode>description_error</InlineCode> и{" "}
              <InlineCode>important_labels</InlineCode>.
            </P>
            <CodeBlock label="bash">
              {`curl -X POST "${v1}/gethistory" \\\n  -H "X-API-Key: scai_ваш_ключ" \\\n  -H "Content-Type: application/json" \\\n  -d '{}'`}
            </CodeBlock>
          </Endpoint>

          <Endpoint
            method="GET"
            path="/api/v1/history/image?token=..."
            description="Получение изображения"
            v1={v1}
            noAuth
          >
            <P>
              Отдаёт бинарное тело картинки. Параметр{" "}
              <InlineCode>token</InlineCode> берётся из{" "}
              <InlineCode>image_token</InlineCode> в ответах выше. Токен
              подписан и имеет ограниченный срок жизни.
            </P>

            <div className="overflow-x-auto rounded-lg border border-gray-200">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200 bg-gray-50">
                    <th className="px-4 py-2.5 text-left font-semibold text-gray-700">
                      Код
                    </th>
                    <th className="px-4 py-2.5 text-left font-semibold text-gray-700">
                      Причина
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  <tr>
                    <td className="px-4 py-2.5 font-mono text-green-600">200</td>
                    <td className="px-4 py-2.5 text-gray-600">
                      Изображение, Content-Type определяется автоматически
                    </td>
                  </tr>
                  <tr>
                    <td className="px-4 py-2.5 font-mono text-red-600">403</td>
                    <td className="px-4 py-2.5 text-gray-600">
                      Токен невалидный или истёк
                    </td>
                  </tr>
                  <tr>
                    <td className="px-4 py-2.5 font-mono text-red-600">404</td>
                    <td className="px-4 py-2.5 text-gray-600">
                      Файл не найден
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Endpoint>
        </div>
      </Section>

      {/* ---- lifecycle ---- */}
      <Section id="lifecycle" icon={Clock} title="Жизненный цикл задания">
        <div className="flex flex-wrap items-center gap-2 rounded-lg bg-gray-50 px-4 py-3 font-mono text-sm">
          <Badge color="blue">pending</Badge>
          <span className="text-gray-400">&rarr;</span>
          <Badge color="yellow">processing</Badge>
          <span className="text-gray-400">&rarr;</span>
          <Badge color="green">completed</Badge>
          <span className="mx-1 text-gray-300">|</span>
          <Badge color="red">error</Badge>
        </div>
        <ul className="space-y-1.5 text-sm text-gray-600">
          <li>
            <strong className="font-medium text-gray-800">pending</strong> —
            задание принято, ожидает очереди.
          </li>
          <li>
            <strong className="font-medium text-gray-800">processing</strong> —
            модель работает над снимком.
          </li>
          <li>
            <strong className="font-medium text-gray-800">completed</strong> —
            классификация готова в поле <InlineCode>result</InlineCode>; описание
            может ещё догружаться через <InlineCode>description_status</InlineCode>.
          </li>
          <li>
            <strong className="font-medium text-gray-800">error</strong> —
            что-то пошло не так, подробности в{" "}
            <InlineCode>result.detail</InlineCode>.
          </li>
        </ul>
      </Section>

      {/* ---- limits ---- */}
      <Section id="limits" icon={ShieldCheck} title="Лимиты">
        <P>
          API ограничивает количество запросов на пользователя в скользящем окне
          60 секунд (по умолчанию <strong>5 запросов/мин</strong>, точное
          значение зависит от инсталляции).
        </P>
        <CodeBlock label="429 — пример ответа">
          {`{ "detail": "Превышен лимит API: не более 5 запросов в минуту. Повторите позже." }`}
        </CodeBlock>
        <P>Просто подождите и повторите запрос.</P>
      </Section>

      {/* ---- errors ---- */}
      <section className="card-elevated space-y-4">
        <h2 className="text-lg font-semibold text-gray-900">Формат ошибок</h2>
        <P>Все ошибки приходят как JSON:</P>
        <CodeBlock>{`{ "detail": "Текстовое описание проблемы" }`}</CodeBlock>
        <P>
          В редких случаях (ошибки валидации формы) <InlineCode>detail</InlineCode>{" "}
          может быть массивом объектов с полем <InlineCode>msg</InlineCode>.
        </P>
      </section>

      {/* ---- health ---- */}
      <section className="card-elevated space-y-4">
        <h2 className="text-lg font-semibold text-gray-900">
          Проверка доступности
        </h2>
        <P>
          Метод <InlineCode>GET {base}/health</InlineCode> не требует ключа и не
          входит в <InlineCode>/api/v1</InlineCode>. Вернёт{" "}
          <InlineCode>200</InlineCode> с телом:
        </P>
        <CodeBlock>{`{ "status": "ok" }`}</CodeBlock>
        <P>
          <InlineCode>503</InlineCode> — сервис временно недоступен.
        </P>
      </section>

      {/* ---- full example ---- */}
      <section className="card-elevated space-y-4">
        <h2 className="text-lg font-semibold text-gray-900">
          Полный пример
        </h2>
        <P>Загрузка, ожидание результата и получение истории — одним скриптом:</P>
        <CodeBlock label="bash">
          {`BASE="${base}"
KEY="scai_xxxxxxxxxxxxxxxx"

# Загрузка
RESP=$(curl -sS -X POST "$BASE/api/v1/uploadfile" \\
  -H "X-API-Key: $KEY" \\
  -F "file=@./sample.jpg")
echo "$RESP"

JOB_ID=$(echo "$RESP" | grep -o '"job_id":[0-9]*' | grep -o '[0-9]*')

# Ожидание результата
while true; do
  STATUS=$(curl -sS "$BASE/api/v1/classification-jobs/$JOB_ID" \\
    -H "X-API-Key: $KEY")
  echo "$STATUS"
  echo "$STATUS" | grep -q '"status":"error"' && break
  echo "$STATUS" | grep -q '"status":"completed"' && \\
    echo "$STATUS" | grep -q '"description_status":"completed"\\|"description_status":"error"\\|"description_status":null' && break
  sleep 2
done

# История
curl -sS -X POST "$BASE/api/v1/gethistory" \\
  -H "X-API-Key: $KEY" \\
  -H "Content-Type: application/json" \\
  -d '{}'`}
        </CodeBlock>
      </section>

      <footer className="text-center text-sm text-gray-400 pt-4">
        Skin Cancer AI &mdash; подробнее на{" "}
        <a
          href="https://skin-cancer-ai.ru"
          className="text-link"
          target="_blank"
          rel="noreferrer"
        >
          skin-cancer-ai.ru
        </a>
      </footer>
    </div>
  )
}

const Endpoint = ({ method, path, description, children, noAuth }) => (
  <div className="space-y-3 rounded-lg border border-gray-200 bg-white p-5">
    <div className="flex flex-wrap items-center gap-2">
      <MethodBadge method={method} />
      <code className="text-sm font-semibold text-gray-800">{path}</code>
      {noAuth && (
        <span className="text-xs text-gray-400">(без ключа)</span>
      )}
    </div>
    <p className="text-sm font-medium text-gray-700">{description}</p>
    {children}
  </div>
)

const Callout = ({ children }) => (
  <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
    {children}
  </div>
)

export default ApiDocs
