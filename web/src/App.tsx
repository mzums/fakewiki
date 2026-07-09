import { Routes, Route } from 'react-router-dom'
import Home from './pages/Main_Page'
import All_Articles from './pages/All_Articles'
import Article from './pages/Article'
//import Article2 from './pages/Article2'
import Layout from "./Layout";

function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Home />} />
        <Route path="/All_Articles" element={<All_Articles />} />
        <Route path="/Article/:title" element={<Article />} />
        <Route path="/Article/Albert Einstein" element={<Article />} />
        <Route path="/Article/XYZ" element={<Article />} />
      </Route>
    </Routes >
  )
}

export default App