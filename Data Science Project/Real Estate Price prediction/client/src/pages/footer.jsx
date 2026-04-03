import Logo from "../assets/vector.png";
import { Link } from "react-router-dom";

const Footer = () => {
    return (
        <div>
             <div className="flex justify-between p-5">
               <div>
                <img src={Logo} alt="Logo" />
                <p>Our ML app help you for descion making, one click at time</p>
               </div>

               <div>
                   <span>About</span>
                   <Link>Our Story</Link>
                   <Link>Carrers</Link>
                   <Link>Our Team</Link>
                   <Link>Resources</Link>
               </div>

               <div>
                <span>Support</span>
                <Link>FAQ</Link>
                <Link>Contact Us</Link>
                <Link>Help Center</Link>
                <Link>Terms of Service</Link>
               </div>

               <div>
                <span>Find Us</span>
                <Link>Events</Link>
                <Link>Locations</Link>
                <Link>Newsletter</Link>
               </div>

               <div>
                <span>Our Social</span>
                <Link>Instagram</Link>
                <Link>Facebook</Link>
                <Link>Twitter</Link>
               </div>
             </div>
        </div>
    )
}

export default Footer;