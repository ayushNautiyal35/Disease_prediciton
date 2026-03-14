import axios from "axios"
// function getToken() {
//     // getting stored value
//     const saved = localStorage.getItem(docDetails);
//     const initial = JSON.parse(saved.Token);
//     return initial || "";
// }
// axios.defaults.headers.common = {'Authorization': `bearer ${getToken()}`}
export default axios.create({
    baseURL: 'https://disease-prediction-backend-ex9e.onrender.com/api'
});