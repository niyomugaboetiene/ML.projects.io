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
             <section id='home'>
                <HomeComponent />
            </section>
    
           <section id='index'>
               <IndexComponent />
           </section>

           <section id='service'>
            <Service />
           </section>

           <section id='about'>
            <About />
           </section>
       </Layout>
    </BrowserRouter>

  )
}

export default App
