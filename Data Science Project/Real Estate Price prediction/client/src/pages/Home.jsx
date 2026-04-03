import { Link } from "react-router-dom";
import vector from "../assets/vector.jpg";

const HomeComponent = () => {


    return (
        <div>
            <div>
               <div>
                <img src={vector} alt="Logo" />
               </div>

                <nav>
                   <Link>Home</Link>
                   <Link>Service</Link>
                   <Link>Contact</Link>
                </nav>

                <button>Sign Up</button>
            </div>
        </div>
    )
}

export default HomeComponent;