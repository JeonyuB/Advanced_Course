import './App.css'
import {Routes, Route} from "react-router-dom";
import ProductPage from "./pages/ProductPage.jsx";
import SalesPage from "./pages/SalesPage.jsx";
import CategoryPage from "./pages/CategoryPage.jsx";
import MainPagePage from "./pages/MainPagePage.jsx";

function App() {

  return (
    <Routes>
        <Route path="/" element={<MainPagePage />} />
        <Route path="/product" element={<ProductPage/>} />
        <Route path="/sales" element={<SalesPage/>} />
        <Route path="/category" element={<CategoryPage/>} />
    </Routes>
  )
}

export default App
