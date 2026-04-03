import axios from "axios";
import { useState, useEffect } from "react";

const IndexComponent = () => {
    const [location, setLocation] = useState(null);
    const [predicted_result, setResult] = useState("");
    const [total_sqft, setTotal_Sqft] = useState(0);
    const [bhk, setBhk] = useState(0);
    const [bath, setBath] = useState(0);

    useEffect(() => {
         const GetLocationNames = async() => {
               try {
                   const res = await axios.get("http://127.0.0.1:5000/get_location_names");
                   setLocation(res.data.locations);
                //    console.log("Locations", res.data.locations);
               } catch (err) {
                   console.error(err);
                   process.exit(1);
               }
        }

        GetLocationNames();
    }, []);


    const HandlePredict = async () => {
        try {
            const res = await axios.post('http://127.0.0.1:5000/predict_home_price', {
                location, total_sqft, bhk, bath
            });
          
            setResult(res.data.estimated_price);
            
        } catch (err){
           console.error(err);
        }
    }


    return (
        <div className="flex bg-emerald-50 justify-center items-center h-screen">
            <div className="bg-emerald-300 p-6 rounded-lg shadow-xl w-1/4">
                 <h1 className="text-center text-xl text-gray-700 font-bold">House Details</h1>
                <div className="mt-2">
                    <label htmlFor="" className="block text-gray-700 text-lg">BHK (Bathroom | Hall | Kitchen)</label>
                    <input type="number"  onChange={(e) => setBhk(e.target.value)}  required
                       className="border w-full py-2 rounded-lg border-gray-700 px-2"
                    />
                </div>
              
                <div className="mt-2">
                    <label htmlFor="" className="block text-gray-700 text-lg">Bathroom</label>
                    <input type="number"  onChange={(e) => setBath(e.target.value)} required
                        className="border w-full py-2 rounded-lg border-gray-700 px-2"
                    />
                </div>
                
                <div className="mt-2">
                    <label htmlFor="" className="block text-gray-700 text-lg">Square feet</label>
                    <input type="number"  onChange={(e) => setTotal_Sqft(e.target.value)}  required
                        className="border w-full py-2 rounded-lg border-gray-700 px-2"
                    />
                </div>
                
                <div className="mt-2">
                    <label htmlFor="" className="block text-gray-700 text-lg">Location</label>
                    <select className="border w-full py-2 rounded-lg border-gray-700" required>
                        {location?.map((loc, idx) => (
                             <option value={`${loc}`} key={idx}>{loc}</option>
                        ))}
                    </select>
                </div>

                <button onClick={HandlePredict} className="w-full mt-3 bg-emerald-500 py-2 rounded-lg text-white font-bold hover:bg-emerald-600 transition-colors">Get Price</button>

                <h2>{predicted_result}</h2>
            </div>
        </div>
    )

}

export default IndexComponent;