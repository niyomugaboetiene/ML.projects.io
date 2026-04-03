import axios from "axios";
import { useState, useEffect } from "react";

const IndexComponent = () => {
    const [location, setLocation] = useState(null);
    const [predicted_result, setResult] = useState("");
    const [sqft, setSqft] = useState(0);
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


    const HandlePredict = async() => {
        try {
            const res = await axios.post('http://127.0.0.1:5000/predict_home_price', {
                location, sqft, bhk, bath
            });
          
            setResult(res.data.estimated_price);
            
        } catch (err){
           console.error(err);
        }
    }


    return (
        <div>
            <div>
                 <h1>House Details</h1>
                <div>
                    <label htmlFor="">BHK (Bathroom | Hall | Kitchen)</label>
                    <input type="number"  onChange={(e) => setBhk(e.target.value)} />
                </div>
              
                <div>
                    <label htmlFor="">Bathroom</label>
                    <input type="number"  onChange={(e) => setBath(e.target.value)} />
                </div>
                
                <div>
                    <label htmlFor="">Square feet</label>
                    <input type="number"  onChange={(e) => setSqft(e.target.value)} />
                </div>
                
                <div>
                    <label htmlFor="">Location</label>
                    <select name="" id="">
                        {location?.map((loc, idx) => (
                             <option value={`${loc}`} key={idx}>{loc}</option>
                        ))}
                    </select>
                </div>

                <button onClick={HandlePredict}>Get Price</button>
            </div>
        </div>
    )

}

export default IndexComponent;