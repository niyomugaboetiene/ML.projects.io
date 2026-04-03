import { Link } from "react-router-dom";
import vector from "../assets/vector.png";

const HomeComponent = () => {


    return (
        <div className="">
            <div className="bg-stone-950  flex p-3 justify-between shadow-lg">
                <img src={vector} alt="Logo" className="w-25 relative left-4 hover:scale-150 transition duration-200" title="Logo"/>
                <nav className="justify-between space-x-24 flex text-stone-400 font-serif items-center text-[18px]">
                   <Link className="hover:text-stone-500 transition-colors">Home</Link>
                   <Link className="hover:text-stone-500 transition-colors">Service</Link>
                   <Link className="hover:text-stone-500 transition-colors">About</Link>
                   <Link className="hover:text-stone-500 transition-colors">Contact</Link>
                </nav>

             <div>
                <button className="text-stone-400 relative right-5 top-7 bg-stone-600 px-6 py-2 rounded-lg hover:bg-stone-700 transition-colors">Sign Up</button>

             </div>
            </div>

            <div>
                <div>
                    <h1>Predict Best Price Of Dream House</h1>

                  <p>Explore our smart house price prediction system that helps you estimate property values based on location, size, and features. Make better decisions with data-driven insights.</p>
                </div>
            </div>
        </div>
    )
}

export default HomeComponent;