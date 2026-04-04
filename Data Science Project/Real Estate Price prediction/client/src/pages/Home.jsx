import { Link } from "react-router-dom";
import vector from "../assets/vector.png";
import { useNavigate } from "react-router-dom";
import house from "../assets/house.jpg";

const HomeComponent = () => {
    const navigate = useNavigate();

    return (
        <div className="bg-stone-950 h-screen">
            <div className="fixed top-0 right-0 left-0 bg-stone-950  flex p-3 justify-between shadow-lg  border-b border-orange-600 z-50">
                <img src={vector} alt="Logo" className="w-25 relative left-4 hover:scale-150 transition duration-200" title="Logo"/>
                <nav className="justify-between space-x-24 flex text-amber-700 font-serif items-center text-[18px]">
                   <button 
                      className="hover:text-amber-800 transition-colors" 
                      onClick={() => document.getElementById("home").scrollIntoView({ behavior: "smooth" })}
                   >
                      Home
                  </button>
                   <button 
                      className="hover:text-amber-800 transition-colors"
                      onClick={() => document.getElementById("service").scrollIntoView({ behavior: "smooth" })}
                   >
                    Service
                  </button>
                   <button 
                      onClick={() => document.getElementById("about").scrollIntoView({ behavior: 'smooth' })}
                      className="hover:text-amber-800 transition-colors"
                    >
                        About
                  </button>
                </nav>

             <div>
                <a 
                   className="font-serif relative right-5 top-7 bg-amber-600 px-6 py-2 text-white  rounded-lg hover:bg-amber-700 transition-colors"
                   href="https://niyomugaboetiene.vercel.app/#contact" rel="noopener noreferrer" target="_blank"

                >
                    Contact
                </a>

             </div>
            </div>

            <div className="flex  mt-12">
                <div className="w-130 p-4 relative">
                    <h1 className="text-[30px] absolute top-70 left-20 font-bold text-orange-500 font-serif">Predict Best Price Of Your Dream House</h1>

                  <p className="text-[20px] absolute top-95 left-20 text-orange-400 font-serif">Your <span className="font-bold">AI</span> tool to explore our smart house price prediction system that helps you estimate property values based on location, size, and features. Make better decisions with data-driven insights.</p>
                    
                
                <div className="text-center absolute top-120 left-17">
                         <h2 className="text-xl mb-4">Ready to explore?</h2>
                        <button onClick={() => document.getElementById("index").scrollIntoView({ behavior: "smooth" })}
                        className="bg-orange-500 px-6 py-2 rounded-lg hover:bg-orange-600 transition-colors font-serif text-white">
                          Try Prediction
                         </button>
                       </div>
                </div>

                <img src={house} alt="House" title="House" className="w-250 rounded-full relative -right-90 top-30"/>
            </div>
        </div>
    )
}

export default HomeComponent;