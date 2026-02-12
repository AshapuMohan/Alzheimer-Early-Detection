
import React from 'react';
import { Link } from 'react-router-dom';
import { BeakerIcon } from '@heroicons/react/24/solid';

const Navbar = () => (
  <nav className="fixed top-0 left-0 right-0 z-50 bg-white/90 backdrop-blur-md border-b border-slate-100 h-16 flex items-center shadow-sm transition-all duration-300">
    <div className="max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 flex items-center justify-between">

      {/* Brand Logo */}
      <Link to="/" className="flex items-center gap-2 group">
        <div className="w-8 h-8 rounded-lg bg-teal-50 flex items-center justify-center text-teal-600 group-hover:bg-teal-600 group-hover:text-white transition-colors duration-300">
          <BeakerIcon className="w-5 h-5" />
        </div>
        <span className="text-xl font-bold tracking-tight text-slate-900 font-['Outfit']">
          Alzheimer<span className="text-teal-600">AI</span>
        </span>
      </Link>

      {/* Navigation Links */}
      <ul className="hidden md:flex items-center space-x-8">
        <li>
          <Link
            to="/"
            className="text-sm font-medium text-slate-600 hover:text-teal-600 transition-colors font-['Inter']"
          >
            Home
          </Link>
        </li>
        <li>
          <Link
            to="/upload"
            className="text-sm font-medium text-slate-600 hover:text-teal-600 transition-colors font-['Inter']"
          >
            Upload MRI
          </Link>
        </li>
        <li>
          <Link
            to="/about"
            className="text-sm font-medium text-slate-600 hover:text-teal-600 transition-colors font-['Inter']"
          >
            Methodology
          </Link>
        </li>
      </ul>

      {/* CTA Button */}
      <div className="flex items-center">
        <Link
          to="/upload"
          className="hidden sm:inline-flex items-center justify-center px-5 py-2 text-sm font-medium text-white transition-all duration-200 bg-teal-600 rounded-full hover:bg-teal-700 hover:shadow-md hover:shadow-teal-600/20 active:scale-95"
        >
          Get Started
        </Link>
      </div>

    </div>
  </nav>
);

export default Navbar;
