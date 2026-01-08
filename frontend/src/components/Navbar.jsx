import React from 'react';
import { Link } from 'react-router-dom';


const Navbar = () => (
  <nav className="fixed lg:mx-[150px] rounded-full top-0 left-0 right-0 z-50 h-[50px] bg-white/70 backdrop-blur-md border border-gray-200 flex items-center justify-between py-1 px-8 shadow-sm">
    <div className="w-full flex items-center justify-between">
      <Link to="/" className="text-2xl font-bold tracking-wide text-gray-800 hover:text-blue-500 transition-colors">
        Alzheimer<span className="text-blue-500">AI</span>
      </Link>
      <ul className="flex items-center space-x-8">
        <li><Link className="text-sm font-medium text-gray-900 hover:text-blue-500 transition-colors" to="/">Home</Link></li>
        <li><Link className="text-sm font-medium text-gray-900 hover:text-blue-500 transition-colors" to="/upload">Upload MRI</Link></li>
        <li><Link className="text-sm font-medium text-gray-900 hover:text-blue-500 transition-colors" to="/about">About</Link></li>
      </ul>
    </div>
  </nav>
);

export default Navbar;
