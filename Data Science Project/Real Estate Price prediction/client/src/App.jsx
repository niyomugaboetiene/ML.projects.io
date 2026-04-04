import { useState } from 'react'
import IndexComponent from './pages/index';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './pages/layout';
import Service from './pages/Service';
import HomeComponent from './pages/Home';
import About from './pages/About';

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

           <section>
            <Service />
           </section>

           <section>
            <About />
           </section>
       </Layout>
    </BrowserRouter>

  )
}

export default App
