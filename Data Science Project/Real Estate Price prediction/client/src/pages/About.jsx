import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faChartLine, faEye, faListCheck } from '@fortawesome/free-solid-svg-icons'

function About() {
  return (
    <div className="min-h-screen bg-stone-950 text-white px-6 py-16">

      {/* TITLE */}
      <div className="text-center mb-16">
        <h1 className="text-4xl font-bold text-orange-500">About Our App</h1>
        <p className="mt-4 text-gray-400 max-w-xl mx-auto">
          Discover how our intelligent system helps you estimate house prices with accuracy and confidence.
        </p>
      </div>

      {/* SECTION 1 */}
      <div className="max-w-5xl mx-auto mb-20 text-center">
        <h2 className="text-2xl font-semibold mb-4 text-orange-400">What We Do</h2>
        <p className="text-gray-300">
          Our machine learning-based platform predicts house prices using key features like location,
          square footage, number of rooms, and bathrooms. It helps buyers, sellers, and investors make
          smarter real estate decisions.
        </p>
      </div>

      {/* FEATURES */}
      <div className="grid md:grid-cols-3 gap-10 max-w-6xl mx-auto mb-20">

        {/* Feature 1 */}
        <div className="bg-stone-900 p-6 rounded-xl shadow-lg text-center hover:scale-105 transition">
          <FontAwesomeIcon icon={faChartLine} className="text-orange-500 text-3xl mb-4" />
          <h3 className="text-xl font-semibold mb-2">Accurate Prediction</h3>
          <p className="text-gray-400">
            Uses trained machine learning models to estimate property prices based on real data.
          </p>
        </div>

        {/* Feature 2 */}
        <div className="bg-stone-900 p-6 rounded-xl shadow-lg text-center hover:scale-105 transition">
          <FontAwesomeIcon icon={faEye} className="text-orange-500 text-3xl mb-4" />
          <h3 className="text-xl font-semibold mb-2">Transparent Process</h3>
          <p className="text-gray-400">
            Every prediction is based on visible input values — no hidden logic, just smart calculations.
          </p>
        </div>

        {/* Feature 3 */}
        <div className="bg-stone-900 p-6 rounded-xl shadow-lg text-center hover:scale-105 transition">
          <FontAwesomeIcon icon={faListCheck} className="text-orange-500 text-3xl mb-4" />
          <h3 className="text-xl font-semibold mb-2">Simple & Fast</h3>
          <p className="text-gray-400">
            Enter your details and get results instantly with a smooth and user-friendly interface.
          </p>
        </div>

      </div>

      {/* HOW IT WORKS */}
      <div className="max-w-4xl mx-auto text-center mb-20">
        <h2 className="text-2xl font-semibold text-orange-400 mb-4">How It Works</h2>
        <p className="text-gray-300">
          Our system uses a trained Linear Regression model that analyzes patterns in housing data.
          When you enter your inputs, the model processes them and predicts a price based on learned relationships.
        </p>
      </div>

      <div className="text-center">
        <h2 className="text-xl mb-4">Ready to explore?</h2>
        <button className="bg-orange-500 px-6 py-2 rounded-lg hover:bg-orange-600 transition">
          Try Prediction
        </button>
      </div>

    </div>
  )
}

export default About