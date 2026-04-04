import { useState } from 'react'
import IndexComponent from './pages/index';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './pages/layout';
import Service from './pages/Service';
import HomeComponent from './pages/Home';

function App() {

  return (
    <BrowserRouter>
        <Layout>
             <section>
                <HomeComponent />
            </section>
    
           <section>
               <IndexComponent />
           </section>
       </Layout>

      <Routes>
               <Route path='/service' element={ <Service />} />

      </Routes>
    </BrowserRouter>

  )
}

export default App
