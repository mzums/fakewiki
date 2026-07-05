import { Routes, Route } from 'react-router-dom'
import Home from './pages/Main_Page'
import About from './pages/All_Articles'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/about" element={<About />} />
    </Routes>
  )
}

export default App