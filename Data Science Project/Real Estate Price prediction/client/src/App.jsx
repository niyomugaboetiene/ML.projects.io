import { useState } from 'react'
import IndexComponent from './pages/index';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import HomeComponent from './pages/Home';

function App() {
  const [count, setCount] = useState(0)

  return (
       <BrowserRouter>
          <Routes>
            <Route path='/predict' element={<IndexComponent />}/>
            <Route path='/' element={<HomeComponent />}/>
          </Routes>
       </BrowserRouter>
  )
}

export default App
