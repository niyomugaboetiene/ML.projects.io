import { Link } from "react-router-dom";
import vector from "../assets/vector.png";
import { useNavigate } from "react-router-dom";
import house from "../assets/house.jpg";

const HomeComponent = () => {
    const navigate = useNavigate();

    return (
        <div className="bg-stone-950 min-h-screen">
            <div className="fixed top-0 right-0 left-0 bg-stone-950 flex p-3 justify-between shadow-lg border-b border-orange-600 z-50">
                <img 
                    src={vector} 
                    alt="Logo" 
                    className="md:w-20 w-16 relative left-2 md:left-4 hover:scale-150 transition duration-200" 
                    title="Logo"
                />
                
                <nav className="hidden md:flex justify-between space-x-8 lg:space-x-24 w-auto md:w-2/3 lg:w-1/2 text-amber-700 font-serif items-center text-sm md:text-base lg:text-[18px]">
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
                       className="font-serif relative right-2 md:right-5 top-5 md:top-7 bg-amber-600 px-4 md:px-6 py-1 md:py-2 text-white rounded-lg hover:bg-amber-700 transition-colors text-sm md:text-base"
                       href="https://niyomugaboetiene.vercel.app/#contact" 
                       rel="noopener noreferrer" 
                       target="_blank"
                    >
                        Contact
                    </a>
                </div>
            </div>

            <div className="flex flex-col md:flex-row mt-12 md:mt-12 px-4 md:px-0">
                <div className="w-full md:w-[500px] lg:w-[600px] p-4 relative min-h-[500px] md:min-h-[600px]">
                    <h1 className="text-2xl sm:text-3xl md:text-[30px] absolute top-20 md:top-40 lg:top-70 left-4 md:left-10 lg:left-20 font-bold text-orange-500 font-serif leading-tight">
                        Predict Best Price Of Your Dream House
                    </h1>

                    <p className="text-base sm:text-lg md:text-[20px] absolute top-48 md:top-60 lg:top-95 left-4 md:left-10 lg:left-20 text-orange-400 font-serif">
                        Your <span className="font-bold">AI</span> tool to explore our smart house price prediction system that helps you estimate property values based on location, size, and features. Make better decisions with data-driven insights.
                    </p>
                    
                    <div className="text-center absolute top-80 md:top-96 lg:top-120 left-0 md:left-4 lg:left-17 w-full md:w-auto">
                        <h2 className="text-lg md:text-xl mb-4 text-white">Ready to explore?</h2>
                        <button 
                            onClick={() => document.getElementById("index").scrollIntoView({ behavior: "smooth" })}
                            className="bg-orange-500 px-6 py-2 rounded-lg hover:bg-orange-600 transition-colors font-serif text-white text-sm md:text-base"
                        >
                            Try Prediction
                        </button>
                    </div>
                </div>

                <div className="w-full md:w-auto flex justify-center md:block mt-8 md:mt-0">
                    <img 
                        src={house} 
                        alt="House" 
                        title="House" 
                        className="w-full max-w-md md:max-w-lg lg:max-w-2xl rounded-2xl md:rounded-full relative md:-right-10 lg:-right-90 md:top-10 lg:top-30 shadow-2xl"
                    />
                </div>
            </div>

      </div>
    )
}

export default HomeComponent;