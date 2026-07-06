import { Routes, Route } from 'react-router-dom'
import Home from './pages/Main_Page'
import All_Articles from './pages/All_Articles'
import Layout from "./Layout";

function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Home />} />
        <Route path="/All_Articles" element={<All_Articles />} />
      </Route>
    </Routes >
  )
}

export default App