import React from "react"

const SignIn = () => {
    return (
        <div className="flex min-h-screen items-center justify-center bg-gray-100 p-6">
            <div className="w-full max-w-md rounded-2xl bg-white p-8 shadow-lg">
                <h2 className="mb-6 text-center text-2xl font-semibold text-gray-700">Sign In</h2>
                <form className="space-y-4">

                    <input
                        type="text"
                        placeholder="Login" 
                        className="w-full rounded-lg border border-gray-300 p-3 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                    />

                    <input 
                        type="password" 
                        placeholder="Password" 
                        className="w-full rounded-lg border border-gray-300 p-3 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                    />

                    <button 
                        type="submit" 
                        className="w-full rounded-lg bg-blue-600 px-4 py-3 text-white font-semibold transition hover:bg-blue-700"
                    >
                        Sign In
                    </button>

                </form>
            </div>
        </div>
    );
};

export default SignIn;