import React from "react"
import ReactDOM from "react-dom/client"
import { Provider } from "react-redux"
import { BrowserRouter as Router, createBrowserRouter, RouterProvider } from "react-router-dom"
import App from "./components/App"
import store from "./store/index"
import "./styles/index.css"

const router = createBrowserRouter(
  [
    {
      path: "/",
      element: <App />,
    }
  ],
  {
    future: {
      v7_startTransition: true,
      v7_relativeSplatPath: true
    }
  }
);

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Provider store={store}>
      <Router>
          <App />
      </Router>
    </Provider>
  </React.StrictMode>,
)