import axios from "axios";
import { useState, useEffect } from "react";

const InputField = ({ label, sublabel, onChange, icon }) => (
  <div className="mb-5">
    <div className="flex justify-between items-baseline mb-1.5">
      <label className="text-xs tracking-widest uppercase text-amber-700 font-medium">
        {label}
      </label>
      {sublabel && (
        <span className="text-xs text-stone-600 tracking-wide">{sublabel}</span>
      )}
    </div>
    <div className="relative">
      {icon && (
        <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-stone-500 text-sm pointer-events-none">
          {icon}
        </span>
      )}
      <input
        type="number"
        onChange={(e) => onChange(e.target.value)}
        required
        className={`w-full bg-stone-950 border border-stone-800 rounded-lg py-3 pr-4 text-stone-200 text-base outline-none transition-colors duration-200 focus:border-amber-700 placeholder-stone-700 [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none ${icon ? "pl-9" : "pl-4"}`}
      />
    </div>
  </div>
);

const IndexComponent = () => {
  const [locations, setLocations] = useState([]);
  const [selectedLocation, setSelectedLocation] = useState("");
  const [predicted_result, setResult] = useState("");
  const [total_sqft, setTotal_Sqft] = useState("");
  const [bhk, setBhk] = useState("");
  const [bath, setBath] = useState("");
  const [loading, setLoading] = useState(false);
  const BACKEND_URL = import.meta.env.BACKEND_URL;

  useEffect(() => {
    const GetLocationNames = async () => {
      try {
        const res = await axios.get(`${BACKEND_URL}/get_location_names`);
        setLocations(res.data.locations);
        if (res.data.locations?.length > 0) setSelectedLocation(res.data.locations[0]);
      } catch (err) {
        console.error(err);
      }
    };
    GetLocationNames();
  }, []);

  const HandlePredict = async () => {
    setLoading(true);
    try {
      const res = await axios.post(`${BACKEND_URL}/predict_home_price`, {
        location: selectedLocation,
        total_sqft,
        bhk,
        bath,
      });
      setResult(res.data.estimated_price);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-stone-950 flex items-center justify-center px-4 font-serif">

      <div className="w-full max-w-sm bg-stone-900 border border-stone-800 rounded-2xl px-8 py-9 relative overflow-hidden">

        <div className="absolute top-0 left-[10%] right-[10%] h-px bg-linear-to-r from-transparent via-amber-700 to-transparent" />

        <div className="text-center mb-8">
          <p className="text-xs tracking-[0.2em] font-mono uppercase text-amber-700 mb-2">Bengaluru Real Estate Valuation</p>
          <h1 className="text-xl text-stone-200 font-normal tracking-wide">Price Estimator</h1>
          <div className="w-8 h-px bg-stone-700 mx-auto mt-3" />
        </div>

        <InputField label="BHK" sublabel="Bed · Hall · Kitchen" onChange={setBhk} icon="⌂" />
        <InputField label="Bathrooms" onChange={setBath} icon="◈" />
        <InputField label="Area" sublabel="sq. ft." onChange={setTotal_Sqft} icon="▣" />

        <div className="mb-7">
          <label className="block text-xs tracking-widest uppercase text-amber-700 font-medium mb-1.5">
            Location
          </label>
          <div className="relative">
            <select
              value={selectedLocation}
              onChange={(e) => setSelectedLocation(e.target.value)}
              required
              className="w-full bg-stone-950 border border-stone-800 rounded-lg py-3 pl-4 pr-9 text-stone-200 text-sm outline-none appearance-none cursor-pointer transition-colors duration-200 focus:border-amber-700"
            >
              {locations.map((loc, idx) => (
                <option value={loc} key={idx} className="bg-stone-900">{loc}</option>
              ))}
            </select>
            <span className="absolute right-3.5 top-1/2 -translate-y-1/2 text-stone-500 text-xs pointer-events-none">▾</span>
          </div>
        </div>

        <button
          onClick={HandlePredict}
          disabled={loading}
          className={`w-full py-3.5 rounded-lg text-xs tracking-[0.18em] uppercase font-bold transition-all duration-200 ${
            loading
              ? "bg-stone-800 text-stone-600 cursor-not-allowed"
              : "bg-amber-700 text-stone-950 hover:bg-amber-600 active:scale-[0.98]"
          }`}
        >
          {loading ? "Estimating…" : "Estimate Value"}
        </button>

        {predicted_result && (
          <div className="mt-6 p-5 bg-stone-950 border border-stone-800 rounded-xl text-center animate-[fadeIn_0.4s_ease]">
            <p className="text-xs tracking-[0.16em] uppercase text-stone-600 mb-2">Estimated Value</p>
            <p className="text-2xl text-amber-600 tabular-nums tracking-wide">
              ₹{Number(predicted_result).toLocaleString("en-IN")}
            </p>
            <p className="text-xs text-stone-700 tracking-widest mt-1.5 uppercase">Indian Rupees</p>
          </div>
        )}
        <div className="absolute bottom-0 left-[10%] right-[10%] h-px bg-gradient-to-r from-transparent via-stone-700 to-transparent" />
      </div>

      <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
};

export default IndexComponent;