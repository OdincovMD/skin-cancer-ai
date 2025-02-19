import React from 'react';

const SignUp = () => {
  return (
    <div className='form-container'>
       <form className='form'>
            <input 
                type="name" 
                placeholder="Your name"
                className="input"
            />
            <input 
                type="email" 
                placeholder="Email address"
                className="input"
            />
            <input 
                type="password" 
                placeholder="Password"
                className="input"
            />
            <input 
                type="password" 
                placeholder="Confirm password"
                className="input"
            />

            <div className='marketing'>
                <input
                    id="allowEmail"
                    type="checkbox"
                />
                <label htmlFor="allowEmail">I want to join the newsletter</label>
            </div>
            <button className='submit'>
                Sign Up
            </button>
        </form>
    </div>
  )
}

export default SignUp;