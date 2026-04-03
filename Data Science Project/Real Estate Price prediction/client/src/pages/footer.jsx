import Logo from "../assets/vector.png";
import { Link } from "react-router-dom";

const Footer = () => {
    return (
        <div className="bg-linear-to-br from-orange-400 via-red-500 to-purple-600">
             <div className="w-[1700px] flex ms-20 rounded-xl bg-white/20 backdrop-blur-md shadow-lg">
               <div>
                <img src={Logo} alt="Logo" />
                <p className="text-white">Our AI app help you for descion making, one click at time</p>
               </div>

              <div className="flex space-x-12 mt-25">
               <div className="flex flex-col space-x-3 text-white">
                   <span>About</span>
                   <Link>Our Story</Link>
                   <Link>Carrers</Link>
                   <Link>Our Team</Link>
                   <Link>Resources</Link>
               </div>

               <div className="flex flex-col text-white">
                    <span>Support</span>
                    <Link>FAQ</Link>
                    <Link>Contact Us</Link>
                    <Link>Help Center</Link>
                    <Link>Terms of Service</Link>
               </div>

               <div className="flex flex-col text-white">
                     <span>Find Us</span>
                     <Link>Events</Link>
                     <Link>Locations</Link>
                     <Link>Newsletter</Link>
               </div>

               <div className="flex flex-col text-white">
                    <span>Our Social</span>
                    <Link>Instagram</Link>
                    <Link>Facebook</Link>
                    <Link>Twitter</Link>
               </div>
             </div>
          </div>
        </div>
    )
}

export default Footer;