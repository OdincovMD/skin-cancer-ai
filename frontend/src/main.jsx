import React from "react"
import ReactDOM from "react-dom/client"
import { Provider } from "react-redux"
import { BrowserRouter as Router } from "react-router-dom"
import App from "./components/App"
import store from "./store/index"
import { onPageReload } from "./store/userReducer"
import "./styles/index.css"

// До первого рендера: иначе дочерние useEffect (например аватар в профиле) успевают
// выполниться с accessToken === null, пока App ещё не вызвал onPageReload в своём effect.
store.dispatch(onPageReload())

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Provider store={store}>
      <Router>
        <App />
      </Router>
    </Provider>
  </React.StrictMode>,
)
