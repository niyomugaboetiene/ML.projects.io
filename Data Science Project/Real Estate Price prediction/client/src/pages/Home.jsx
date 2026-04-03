import { Link } from "react-router-dom";
import vector from "../assets/vector.png";

const HomeComponent = () => {


    return (
        <div className="bg-stone-950 shadow-lg">
            <div className="flex p-3 justify-between ">
                <img src={vector} alt="Logo" className="w-25 "/>
                <nav className="justify-between space-x-24 flex text-stone-400 font-serif items-center text-[18px]">
                   <Link className="hover:text-stone-500 transition-colors">Home</Link>
                   <Link className="hover:text-stone-500 transition-colors">Service</Link>
                   <Link className="hover:text-stone-500 transition-colors">About</Link>
                   <Link className="hover:text-stone-500 transition-colors">Contact</Link>
                </nav>

             <div>
                <button className="text-stone-400">Sign Up</button>

             </div>
            </div>
        </div>
    )
}

export default HomeComponent;