import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {  faLocationDot, faUserPen, faEye, faHandshake } from "@fortawesome/free-solid-svg-icons"


export default function  Service () {
     return (
        <div className="min-h-screen bg-stone-950 text-white p-2 flex justify-center items-center">
             <div className="absolute -bottom-400 w-120">
                <h1 className="text-xl text-center font-serif text-amber-700 font-bold">Why Choose Us</h1>
                 <p className="font-serif text-lg text-amber-800">Evaluating your home price with expertice, integrity, transparent <span className="ms-25"> and unmatched personalized service</span></p>
             </div>

            <div className="flex p-3 flex-row w-300 space-x-3">
                <div className="bg-amber-700 p-3 rounded-lg">
                    <div className="bg-amber-500 p-3 w-1/3 text-center rounded-lg ">
                        <FontAwesomeIcon icon={faLocationDot} className="text-[40px] text-stone-950" />
                    </div>
                    <h3 className="font-serif font-bold text-stone-950 text-lg">Expert Guidance</h3>
                    <p className="text-stone-950 font-serif">Benefit from our team's seasoned expertice for a smooth home price experience</p>
                </div>

                <div className="bg-amber-700 p-3 rounded-lg">
                    <div className="bg-amber-500 p-3 w-1/3 text-center rounded-lg">
                        <FontAwesomeIcon icon={faUserPen} className="text-[40px] text-stone-950"/>
                    </div>
                    <h3 className="font-serif font-bold text-stone-950 text-lg">Personalized Service</h3>
                    <p className="text-stone-950 font-serif">Our service adapt to your unique needs, making your journey stress-free</p>
                </div>
 
               <div className="bg-amber-700 p-3 rounded-lg">
                <div className="bg-amber-500 p-3 w-1/3 text-center rounded-lg">
                    <FontAwesomeIcon icon={faEye} className="text-[40px] text-stone-950" />
                </div>
                   <h3 className="font-serif font-bold text-stone-950 text-lg">Transparent Process</h3>
                  <p className="text-stone-950 font-serif">Stay informed with our clear and honest approach to buying your home</p>
               </div>

                <div className="bg-amber-700 p-3 rounded-lg">
                    <div className="bg-amber-500 p-3 w-1/3 text-center rounded-lg">
                       <FontAwesomeIcon icon={faHandshake} className="text-[40px] text-stone-950"/>
                    </div>
                    <h3 className="font-serif font-bold text-stone-950 text-lg">Exceptional Support</h3>
                    <p className="text-stone-950 font-serif">Providing peace of mind with our responsive and attentive customer service</p>
                </div>
            </div>
        </div>
     )
}