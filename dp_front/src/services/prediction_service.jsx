import axios from "axios"

const conn = axios.create({
    baseURL: "https://disease-prediction-api-98qs.onrender.com/",
});

class PredictionService {
    constructor(){
        this.endpoint = "dp_pred/";
    }

    getRes(input){
        // const data = { params:input };
        console.log('Data sent :',input);
        const req = conn.post(this.endpoint, input);

        return req;
    }
}

export default new PredictionService();