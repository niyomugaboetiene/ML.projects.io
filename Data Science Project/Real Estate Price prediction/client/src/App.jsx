import { useState } from 'react'
import IndexComponent from './pages/index';
function App() {
  const [count, setCount] = useState(0)

  return (
    <IndexComponent />
  )
}

export default App
