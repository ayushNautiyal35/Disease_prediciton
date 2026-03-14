import React, { useState } from 'react';
import './Register.css'
import { Link, useNavigate } from 'react-router-dom';
import { useFormik } from 'formik';
import * as Yup from 'yup';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import doctorService from '../../services/doc_service';

const initialValues = {
  name: "",
  email: "",
  phone: "",
  password: "",
  confirm_password: "",
  price: "",
  qualifications: [""],
  specialization: "",
  timeSlots: [""],
  // image: null,
  address: "",
  about: ""
};

const doctorSchema = Yup.object({
  name: Yup.string()
    .required('Name is required')
    .min(5, 'Name must be at least 5 characters')
    .max(20, 'Name must be at most 20 characters')
    .matches(/^[a-zA-Z\s]+$/, 'Name can only contain letters and spaces'),
  email: Yup.string().email('Invalid email address').required('Email is required'),
  phone: Yup.string()
    .matches(/^\d{10}$/, 'Phone number must be exactly 10 digits')
    .required('Phone number is required'),
  password: Yup.string().min(8, 'Password must be at least 8 characters').required('Password is required'),
  confirm_password: Yup.string()
    .oneOf([Yup.ref('password'), null], 'Passwords must match')
    .required('Please confirm your password'),
  price: Yup.number()
    .typeError('Price must be a number')
    .required('Price is required'),
  qualifications: Yup.array().of(Yup.string().required('Qualification is required')).min(1, 'At least one qualification is required'),
  specialization: Yup.string().required('Specialization is required'),
  timeSlots: Yup.array().of(Yup.string().required('Time slot is required')).min(1, 'At least one time slot is required'),
  address: Yup.string().required('Address is required'),
  about: Yup.string().required('Introduction is required')
});

const DRegister = () => {
  const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
  const navigate = useNavigate();
  const [qualifications, setQualifications] = useState([""]);
  const [timeSlots, setTimeSlots] = useState([""]);

  const showToastMessage = (msg) => {
    if (msg === "success") {
      toast.success('REGISTERED SUCCESSFULLY!', {
        position: toast.POSITION.BOTTOM_CENTER,
        autoClose: 1000,
        pauseOnHover: false,
      });
    } else {
      toast.warning(msg, {
        position: toast.POSITION.BOTTOM_CENTER,
        autoClose: 1000,
        pauseOnHover: false,
      });
    }
  };

  const { values, errors, touched, handleBlur, handleChange, handleSubmit, setFieldValue, isSubmitting, resetForm } =
    useFormik({
      initialValues,
      validationSchema: doctorSchema,
      onSubmit: async (values, actions) => {

        const Data = {
          'username': values.name,
          'email': values.email,
          'phone': values.phone,
          'password1': values.password,
          'password2': values.confirm_password,
          'price': values.price,
          'qualifications': values.qualifications,
          'specialization': values.specialization,
          'timeSlots': values.timeSlots,
          'address': values.address,
          'about': values.about
        }

        try {
          doctorService.register(Data).then(async (res) => {
            console.log("Registered Values:", values);
            actions.resetForm();
            setQualifications([""]);
            setTimeSlots([""]);
            showToastMessage("success");
            await delay(2000); 
            navigate("/doctorlogin")
          }).catch((err) => {
            showToastMessage(err.response.data.message || "An error occurred. Please try again later.");
          });
        } catch (error) {
          showToastMessage("An error occurred. Please try again later.");
          console.error(error);
        }
      },
    });

  const handleAddQualification = () => {
    setQualifications([...qualifications, ""]);
  };

  const handleRemoveQualification = (index) => {
    const newQualifications = [...qualifications];
    newQualifications.splice(index, 1);
    setQualifications(newQualifications);
    setFieldValue('qualifications', newQualifications);
  };

  const handleQualificationChange = (index, event) => {
    const newQualifications = [...qualifications];
    newQualifications[index] = event.target.value;
    setQualifications(newQualifications);
    setFieldValue('qualifications', newQualifications);
  };

  const handleAddTimeSlot = () => {
    setTimeSlots([...timeSlots, ""]);
  };
  

  const handleRemoveTimeSlot = (index) => {
    const newTimeSlots = [...timeSlots];
    newTimeSlots.splice(index, 1);
    setTimeSlots(newTimeSlots);
    setFieldValue('timeSlots', newTimeSlots);
  };

  const handleTimeSlotChange = (index, event) => {
    const newTimeSlots = [...timeSlots];
    newTimeSlots[index] = event.target.value;
    setTimeSlots(newTimeSlots);
    setFieldValue('timeSlots', newTimeSlots);
  };

  return (
    <>
      <ToastContainer />
      <div className="doctor-register">
        <div className="doctor-cont">
          <h2>Doctor Registration</h2>
          <div className="doctor-main">
            <form onSubmit={handleSubmit} encType="multipart/form-data">
              <div>
                <div className="box">
                  <label htmlFor="name">Name:</label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    placeholder="Your name"
                    value={values.name}
                    onChange={handleChange}
                    onBlur={handleBlur}
                  />
                  {errors.name && touched.name && <p className="form-error">{errors.name}</p>}
                </div>
                <div className="box">
                  <label htmlFor="email">Email:</label>
                  <input
                    type="email"
                    id="email"
                    name="email"
                    placeholder="Your email"
                    value={values.email}
                    onChange={handleChange}
                    onBlur={handleBlur}
                  />
                  {errors.email && touched.email && <p className="form-error">{errors.email}</p>}
                </div>
                <div className="box">
                  <label htmlFor="phone">Phone:</label>
                  <input
                    type="number"
                    id="phone"
                    name="phone"
                    placeholder="Your phone"
                    value={values.phone}
                    onChange={handleChange}
                    onBlur={handleBlur}
                  />
                  {errors.phone && touched.phone && <p className="form-error">{errors.phone}</p>}
                </div>
                <div className="box">
                  <label htmlFor="password">Password:</label>
                  <input
                    type="password"
                    id="password"
                    name="password"
                    placeholder="Password"
                    value={values.password}
                    onChange={handleChange}
                    onBlur={handleBlur}
                  />
                  {errors.password && touched.password && <p className="form-error">{errors.password}</p>}
                </div>
                <div className="box">
                  <label htmlFor="confirm_password">Confirm password:</label>
                  <input
                    type="password"
                    id="confirm_password"
                    name="confirm_password"
                    placeholder="Confirm Password"
                    value={values.confirm_password}
                    onChange={handleChange}
                    onBlur={handleBlur}
                  />
                  {errors.confirm_password && touched.confirm_password && <p className="form-error">{errors.confirm_password}</p>}
                </div>
              </div>
              <div>
                <div className="box">
                  <label htmlFor="price">Fees:</label>
                  <input
                    type="text"
                    id="price"
                    name="price"
                    placeholder="Your price"
                    value={values.price}
                    onChange={handleChange}
                    onBlur={handleBlur}
                  />
                  {errors.price && touched.price && <p className="form-error">{errors.price}</p>}
                </div>
                <div className="box">
                  <label htmlFor="qualifications">Qualifications:</label>
                  {qualifications.map((qualification, index) => (
                    <div key={index} className="dynamic-input">
                      <input
                        type="text"
                        id={`qualifications-${index}`}
                        name={`qualifications[${index}]`}
                        placeholder="Qualification"
                        value={qualification}
                        onChange={(event) => handleQualificationChange(index, event)}
                      />
                      {qualifications.length > 1 && (
                        <button type="button" onClick={() => handleRemoveQualification(index)}>
                          -
                        </button>
                      )}
                  <button type="button" onClick={handleAddQualification}>+</button>
                    </div>
                  ))}
                  {errors.qualifications && touched.qualifications && <p className="form-error">{errors.qualifications}</p>}
                </div>
                <div className="box">
                  <label htmlFor="specialization">Specialization:</label>
                  <input
                    type="text"
                    id="specialization"
                    name="specialization"
                    placeholder="Specialization"
                    value={values.specialization}
                    onChange={handleChange}
                    onBlur={handleBlur}
                  />
                  {errors.specialization && touched.specialization && <p className="form-error">{errors.specialization}</p>}
                </div>
                <div className="box">
                  <label htmlFor="timeSlots">Time Slots:</label>
                  {timeSlots.map((timeSlot, index) => (
                    <div key={index} className="dynamic-input">
                      <input
                        type="text"
                        id={`timeSlots-${index}`}
                        name={`timeSlots[${index}]`}
                        placeholder="Time Slot"
                        value={timeSlot}
                        onChange={(event) => handleTimeSlotChange(index, event)}
                      />
                      {timeSlots.length > 1 && (
                        <button type="button" onClick={() => handleRemoveTimeSlot(index)}>
                          -
                        </button>
                      )}
                      <button type="button" onClick={handleAddTimeSlot}>+</button>
                    </div>
                  ))}
                  {errors.timeSlots && touched.timeSlots && <p className="form-error">{errors.timeSlots}</p>}
                </div>
                {/* <div className="box">
                  <label htmlFor="image">Profile Image:</label>
                  <input
                    type="file"
                    id="image"
                    name="image"
                    onChange={(event) => setFieldValue('image', event.currentTarget.files[0])}
                    onBlur={handleBlur}
                  />
                </div> */}
               
              </div>
              <div>
              <div className="box">
                  <label htmlFor="address">Address:</label>
                  <input
                    type="text"
                    id="address"
                    className='rightsidediv'
                    name="address"
                    placeholder="Address"
                    value={values.address}
                    onChange={handleChange}
                    onBlur={handleBlur}
                  />
                  {errors.address && touched.address && <p className="form-error">{errors.address}</p>}
                </div>
                <div className="box">
                  <label htmlFor="about">Introduction:</label>
                  <textarea
                    className='rightsidediv'
                    id="about"
                    name="about"
                    placeholder="Write about yourself"
                    value={values.about}
                    onChange={handleChange}
                    onBlur={handleBlur}
                  />
                  {errors.about && touched.about && <p className="form-error">{errors.about}</p>}
                </div>
                <p>
        Already have an account? <Link className='signlink' to="/doctorlogin">Sign in</Link>{" "}
      </p>
              <button className='btn' type="submit" disabled={isSubmitting}>Register</button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </>
  );
};

export default DRegister;
