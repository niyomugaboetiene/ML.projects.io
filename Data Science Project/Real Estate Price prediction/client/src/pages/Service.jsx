import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faLocationDot, faUserPen, faEye, faHandshake } from "@fortawesome/free-solid-svg-icons";

export default function Service() {
  return (
    <div className="min-h-screen bg-stone-950 text-white px-6 py-16">

      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold font-serif text-orange-500">Why Choose Us</h1>
        <p className="mt-4 text-gray-400 max-w-2xl mx-auto text-lg font-serif">
          Evaluating your home price with expertise, integrity, transparency, and unmatched personalized service.
        </p>
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 max-w-7xl mx-auto">
        
        <div className="bg-stone-900 p-6 rounded-xl shadow-2xl hover:scale-105 transition duration-300 hover:shadow-amber-700">
          <div className="bg-stone-950 w-16 h-16 flex items-center justify-center rounded-full mb-4 mx-auto">
            <FontAwesomeIcon icon={faLocationDot} className="text-white text-2xl" />
          </div>
          <h3 className="text-center font-bold text-white text-lg mb-2">Expert Guidance</h3>
          <p className="text-center text-white font-serif text-md">
            Benefit from our team's seasoned expertise for a smooth home price experience.
          </p>
        </div>

        <div className="bg-stone-900 p-6 rounded-xl shadow-2xl hover:scale-105 transition duration-300 hover:shadow-amber-700">
          <div className="bg-stone-950 w-16 h-16 flex items-center justify-center rounded-full mb-4 mx-auto">
            <FontAwesomeIcon icon={faUserPen} className="text-white text-2xl" />
          </div>
          <h3 className="text-center font-bold text-white text-lg mb-2">Personalized Service</h3>
          <p className="text-center text-white font-serif ">
            Our service adapts to your unique needs, making your journey stress-free.
          </p>
        </div>

        <div className="bg-stone-900 p-6 rounded-xl shadow-2xl hover:scale-105 transition duration-300 hover:shadow-amber-700">
          <div className="bg-stone-950 w-16 h-16 flex items-center justify-center rounded-full mb-4 mx-auto">
            <FontAwesomeIcon icon={faEye} className="text-white text-2xl" />
          </div>
          <h3 className="text-center font-bold text-white text-lg mb-2">Transparent Process</h3>
          <p className="text-center text-white font-serif text-md">
            Stay informed with our clear and honest approach to buying your home.
          </p>
        </div>

        <div className="bg-stone-900 p-6 rounded-xl shadow-2xl hover:scale-105 transition duration-300 hover:shadow-amber-700">
          <div className="bg-stone-950 w-16 h-16 flex items-center justify-center rounded-full mb-4 mx-auto">
            <FontAwesomeIcon icon={faHandshake} className="text-white text-2xl" />
          </div>
          <h3 className="text-center font-bold text-white text-lg mb-2">Exceptional Support</h3>
          <p className="text-center text-white font-serif text-sm">
            Providing peace of mind with our responsive and attentive customer service.
          </p>
        </div>

      </div>

    </div>
  );
}