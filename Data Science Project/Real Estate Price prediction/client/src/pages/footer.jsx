import Logo from "../assets/vector.png";
import { Link } from "react-router-dom";

const Footer = () => {
    return (
        <div className="bg-linear-to-br from-stone-950 via-stone-950 to-stone-900">
             <div className="w-350 flex ms-65 rounded-xl bg-stone-900 backdrop-blur-md shadow-lg">
               <div className="w-75 p-4">
                <img src={Logo} className="w-100" alt="Logo" />
                <p className="text-amber-700 font-serif text-lg text-center -mt-3">Our AI app help you for descion making, one click at time</p>
               </div>

              <div className="flex space-x-30 mt-25 ms-55">
               <div className="flex flex-col space-x-3 text-amber-700 font-serif">
                   <span className="text-xl font-bold mb-3">About</span>
                   <Link className="hover:underline hover:text-md transition duration-500">Our Story</Link>
                   <Link className="hover:underline hover:text-md transition duration-500">Carrers</Link>
                   <Link className="hover:underline hover:text-md transition duration-500">Our Team</Link>
                   <Link className="hover:underline hover:text-md transition duration-500">Resources</Link>
               </div>

               <div className="flex flex-col text-amber-700 font-serif">
                    <span className="text-xl font-bold mb-3">Support</span>
                    <Link className="hover:underline hover:text-md transition duration-500">FAQ</Link>
                    <Link className="hover:underline hover:text-md transition duration-500">Contact Us</Link>
                    <Link className="hover:underline hover:text-md transition duration-500">Help Center</Link>
                    <Link className="hover:underline hover:text-md transition duration-500">Terms of Service</Link>
               </div>

               <div className="flex flex-col text-amber-700 font-serif">
                     <span className="text-xl font-bold mb-3">Find Us</span>
                     <Link className="hover:underline hover:text-md transition duration-500">Events</Link>
                     <Link className="hover:underline hover:text-md transition duration-500">Locations</Link>
                     <Link className="hover:underline hover:text-md transition duration-500">Newsletter</Link>
               </div>

               <div className="flex flex-col  font-serif space-y-2 items-center ">
                    <span className="text-xl text-amber-700 font-bold mb-3">Our Social</span>
                    <a className="bg-linear-to-b from-purple-400 to-pink-600 bg-transparent text-gray-600 transition duration-500" href={`https://www.instagram.com/__net__250`} target="_blank" rel="noopener noreferrer">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M7.75 2C4.678 2 2 4.678 2 7.75v8.5C2 19.322 4.678 22 7.75 22h8.5C19.322 22 22 19.322 22 16.25v-8.5C22 4.678 19.322 2 16.25 2h-8.5zm0 2h8.5C18.216 4 20 5.784 20 7.75v8.5c0 1.966-1.784 3.75-3.75 3.75h-8.5C5.784 20 4 18.216 4 16.25v-8.5C4 5.784 5.784 4 7.75 4zm9.25 1.5a.75.75 0 100 1.5.75.75 0 000-1.5zM12 7a5 5 0 100 10 5 5 0 000-10zm0 2a3 3 0 110 6 3 3 0 010-6z"/>
                    </svg> <span></span>
                    </a>
                    <a className="hover:underline hover:text-md transition duration-500 text-stone-700" href="https://x.com/Niyomugabo_250" rel="noopener noreferrer" target="_blank">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M18.244 2H21l-6.51 7.44L22 22h-6.828l-5.35-6.98L3.9 22H1l6.97-7.97L2 2h6.828l4.86 6.37L18.244 2zm-2.394 18h1.885L8.19 4H6.2l9.65 16z"/>
                    </svg> 
                    </a>
                    <a className="hover:underline hover:text-md transition duration-500 bg-blue-500 text-white" href="https://web.facebook.com/profile.php?id=100090629463936" rel="noopener noreferrer" target="_blank">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
                           <path d="M22 12a10 10 0 10-11.5 9.87v-6.99H8.078V12h2.422V9.797c0-2.39 1.42-3.713 3.596-3.713 1.043 0 2.134.186 2.134.186v2.35h-1.203c-1.185 0-1.553.736-1.553 1.49V12h2.64l-.422 2.88h-2.218v6.99A10 10 0 0022 12z"/>  
                    </svg>
                    </a>
               </div>
             </div>
          </div>
      
         <div className="mt-3">
          <p className="text-lg text-center text-gray-200 font-serif">  &copy; Predictor. Build with ❤️ and ☕ by <a href="https://github.com/niyomugaboetiene" rel="noopener noreferrer" target="_blank" className="text-amber-800 underline font-bold">Etiene Niyomugabo</a></p> 
          </div>

        </div>
    )
}

export default Footer;