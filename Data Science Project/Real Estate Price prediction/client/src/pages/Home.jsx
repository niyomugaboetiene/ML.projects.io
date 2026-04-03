import { Link } from "react-router-dom";
import vector from "../assets/vector.png";
import house from "../assets/house.jpg";

const HomeComponent = () => {


    return (
        <div className="bg-stone-950 h-screen">
            <div className="bg-stone-950  flex p-3 justify-between shadow-lg  border-b border-orange-600">
                <img src={vector} alt="Logo" className="w-25 relative left-4 hover:scale-150 transition duration-200" title="Logo"/>
                <nav className="justify-between space-x-24 flex text-amber-700 font-serif items-center text-[18px]">
                   <Link className="hover:text-amber-800 transition-colors">Home</Link>
                   <Link className="hover:text-amber-800 transition-colors">Service</Link>
                   <Link className="hover:text-amber-800 transition-colors">About</Link>
                   <Link className="hover:text-amber-800 transition-colors">Contact</Link>
                </nav>

             <div>
                <button className="text-stone-800 relative right-5 top-7 bg-amber-600 px-6 py-2 rounded-lg hover:bg-amber-700 transition-colors">Sign Up</button>

             </div>
            </div>

            <div className="flex  mt-12">
                <div className="w-130 p-4 relative">
                    <h1 className="text-[30px] absolute top-40 left-20 font-bold text-orange-500 font-serif">Predict Best Price Of Dream House</h1>

                  <p className="text-[20px] absolute top-65 left-20">Explore our smart house price prediction system that helps you estimate property values based on location, size, and features. Make better decisions with data-driven insights.</p>
                </div>

                <img src={house} alt="House" title="House" className="w-250 rounded-full relative -right-90"/>
            </div>
        </div>
    )
}

export default HomeComponent;