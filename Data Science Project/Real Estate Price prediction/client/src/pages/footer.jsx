import Logo from "../assets/vector.png";
import { Link } from "react-router-dom";

const Footer = () => {
    return (
        <div className="bg-linear-to-br from-stone-950 via-amber-700 to-stone-900">
             <div className="w-425 flex ms-25 rounded-xl bg-white/20 backdrop-blur-md shadow-lg">
               <div className="w-75 p-4">
                <img src={Logo} className="w-100" alt="Logo" />
                <p className="text-white text-lg text-center -mt-3">Our AI app help you for descion making, one click at time</p>
               </div>

              <div className="flex space-x-30 mt-25 ms-55">
               <div className="flex flex-col space-x-3 text-white">
                   <span className="text-xl font-bold mb-3">About</span>
                   <Link className="hover:underline hover:text-md transition duration-500">Our Story</Link>
                   <Link className="hover:underline hover:text-md transition duration-500">Carrers</Link>
                   <Link className="hover:underline hover:text-md transition duration-500">Our Team</Link>
                   <Link className="hover:underline hover:text-md transition duration-500">Resources</Link>
               </div>

               <div className="flex flex-col text-white">
                    <span className="text-xl font-bold mb-3">Support</span>
                    <Link className="hover:underline hover:text-md transition duration-500">FAQ</Link>
                    <Link className="hover:underline hover:text-md transition duration-500">Contact Us</Link>
                    <Link className="hover:underline hover:text-md transition duration-500">Help Center</Link>
                    <Link className="hover:underline hover:text-md transition duration-500">Terms of Service</Link>
               </div>

               <div className="flex flex-col text-white">
                     <span className="text-xl font-bold mb-3">Find Us</span>
                     <Link className="hover:underline hover:text-md transition duration-500">Events</Link>
                     <Link className="hover:underline hover:text-md transition duration-500">Locations</Link>
                     <Link className="hover:underline hover:text-md transition duration-500">Newsletter</Link>
               </div>

               <div className="flex flex-col text-white">
                    <span className="text-xl font-bold mb-3">Our Social</span>
                    <Link className="hover:underline hover:text-md transition duration-500" to="#">Instagram</Link>
                    <Link className="hover:underline hover:text-md transition duration-500">Facebook</Link>
                    <Link className="hover:underline hover:text-md transition duration-500">Twitter</Link>
               </div>
             </div>
          </div>
        </div>
    )
}

export default Footer;