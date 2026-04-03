import { useState } from 'react'
import IndexComponent from './pages/index';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './pages/layout';
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
    </BrowserRouter>

  )
}

export default App
