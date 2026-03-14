import React, { createContext, useEffect, useState } from 'react';
import userService from '../services/user_service';
import { checkUser, checkUserState, clearToken } from '../services/user_service';
import { checkDoc } from '../services/doc_service';
import { checkAdmin } from '../services/admin_service';

export const SiteContext = createContext(null);

export const SiteContextProvider = (props) => {
    const [uid, setUid] = useState('');
    const [role, setRole] = useState('');
    // const [user, setUser] = useState({ user: "blank" });
    const [userState, setUserState] = useState({
        path: "",
        gender: "",
        age: "",
        quesList: {},
        SymList: {},
        SymValList: {},
        predResult:[],
    });

    const logout = () => {
        console.log("called logout");
        clearToken();
        setUid('');
        window.location.replace('https://disease-prediction-frontend-ejut.onrender.com/');
    };

    useEffect(() => {
        const fetchData = async () => {
            const res1 = checkUser();
            const res2 = checkDoc();
            const res3 = checkAdmin();
            const res4 = checkUserState();

            if (res1 !== '') {
                setUid(res1);
                setRole('user');

            } else if (res2 !== '') {
                setUid(res2);
                setRole('doctor')

            } else {
                setUid(res3);
                setRole('admin')
            }

            setUserState(res4)
        };

        // Run fetchData only in the client-side environment
        if (typeof window !== 'undefined') {
            fetchData();
        }
    }, []);

    // useEffect(() => {
    //     if (uid) {
    //         userService.getInfo(uid)
    //             .then((res) => {
    //                 console.log('Uid :', uid);
    //                 console.log('User :', res.data);
    //                 setUser(res.data);
    //             })
    //             .catch((err) => {
    //                 console.log('Error :', err);
    //             });
    //     }
    // }, [uid]);

    const contextValue = {
        uid,
        setUid,
        logout,
        role,
        userState,
        setUserState,
    };

    return (
        <SiteContext.Provider value={contextValue}>
            {props.children}
        </SiteContext.Provider>
    );
};
