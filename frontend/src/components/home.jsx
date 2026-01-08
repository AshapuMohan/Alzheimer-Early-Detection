import React, { useState } from "react";
import { Link } from "react-router-dom";

// A simple arrow icon for the button
const ArrowIcon = () => (
  <svg className="w-5 h-5 ml-2 transition-transform duration-300 group-hover:translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 8l4 4m0 0l-4 4m4-4H3"></path>
  </svg>
);

const Home = () => {
  const [isHovered, setIsHovered] = useState(false);

  const stats = [
    { value: "61M+", label: "Global Cases (2024)" },
    { value: "153M+", label: "Projected Cases (2050)" },
    { value: "1 in 9", label: "Adults Over 65" },
  ];

  const stages = [
    {
      title: "Non-Demented",
      description: "Represents a baseline of healthy cognitive function without significant memory issues.",
      imgSrc: "/NonDemented.png",
    },
    {
      title: "Very Mild Demented",
      description: "Early stage marked by very slight memory lapses that may not be apparent to others.",
      imgSrc: "/VeryMildDemented.png",
    },
    {
      title: "Mild Demented",
      description: "Clear cognitive challenges emerge that can affect daily life, and some assistance may be needed.",
      imgSrc: "/mild-dent.png",
    },
    {
      title: "Moderate Demented",
      description: "Significant memory loss and confusion become prominent; daily assistance is often necessary.",
      imgSrc: "/ModerateDemented.png",
    },
  ];

  return (
    <div className="bg-white text-gray-800 font-sans">
      {/* HERO SECTION */}
      <section className="text-center py-40 px-6 bg-gradient-to-br from-gray-50 to-blue-50">
        <h1 className="text-5xl md:text-6xl font-extrabold mb-4 text-gray-900 tracking-tight">
          Clarity in Complexity
        </h1>
        <p className="text-xl md:text-2xl text-gray-600 max-w-3xl mx-auto">
          Our advanced AI offers a new perspective on cognitive health, analyzing MRI scans to bring potential early signs of Alzheimer's into focus.
        </p>
        <Link to="/upload">
          <button 
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            className="group mt-12 inline-flex items-center px-8 py-4 text-lg bg-blue-600 text-white font-semibold rounded-full shadow-lg hover:bg-blue-700 hover:shadow-xl transform hover:scale-105 transition-all duration-300 ease-in-out"
          >
            Begin Your Analysis
            <ArrowIcon />
          </button>
        </Link>
      </section>

      {/* SECTION 1: How AI Helps */}
      <section className="py-24 px-6 text-center">
        <h2 className="text-4xl font-bold mb-6 text-gray-900">Illuminating the Unseen</h2>
        <p className="max-w-3xl mx-auto text-lg text-gray-600 mb-16">
          Deep learning algorithms detect subtle neurodegenerative patterns often invisible to the naked eye, providing a crucial window for early awareness and planning.
        </p>
        <div className="max-w-4xl mx-auto rounded-xl shadow-2xl overflow-hidden">
          <img
            src="/intro.png"
            alt="An illustration showing an AI analyzing a brain MRI scan"
            className="w-full"
          />
        </div>
      </section>

      {/* STATISTICS SECTION */}
      <section className="py-24 px-6 text-center bg-gray-50">
        <h2 className="text-4xl font-bold mb-12 text-gray-900">A Global Challenge</h2>
        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {stats.map((stat) => (
            <div key={stat.label} className="bg-white p-8 rounded-xl shadow-lg border border-gray-100 text-center">
              <h3 className="text-5xl font-bold text-blue-600">{stat.value}</h3>
              <p className="text-gray-600 mt-3 font-medium">{stat.label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* DEMENTIA STAGES */}
      <section className="py-24 px-6">
        <h2 className="text-4xl font-bold text-center mb-16 text-gray-900">Mapping the Progression</h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 max-w-7xl mx-auto">
          {stages.map((stage) => (
            <div key={stage.title} className="bg-white rounded-lg shadow-lg overflow-hidden flex flex-col transform hover:-translate-y-2 transition-transform duration-300 hover:shadow-xl">
              <img src={stage.imgSrc} alt={stage.title} className="w-full h-48 object-cover"/>
              <div className="p-6 flex flex-col flex-grow">
                <h3 className="text-xl font-bold mb-2 text-gray-900">{stage.title}</h3>
                <p className="text-gray-600 flex-grow">{stage.description}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* CTA BANNER */}
      <div className="py-20 px-6 text-white text-center bg-blue-600">
        <h2 className="text-4xl md:text-5xl font-bold mb-4">Take the Next Step</h2>
        <p className="text-xl max-w-3xl mx-auto opacity-90 mb-8">
          Gain a deeper understanding of your cognitive health. Our secure, confidential analysis is just a click away.
        </p>
        <Link to="/upload">
          <button className="px-8 py-4 text-lg bg-white text-blue-700 font-semibold rounded-full shadow-lg hover:bg-gray-200 transform hover:scale-105 transition-transform duration-300">
            Upload Your MRI
          </button>
        </Link>
      </div>

      {/* SCROLL TO TOP BUTTON */}
      <button
        aria-label="Scroll to top"
        className="fixed bottom-8 right-8 bg-zinc-700 text-white rounded-full w-14 h-14 flex items-center justify-center shadow-2xl hover:bg-zinc-700 transition-all duration-300 z-50"
        onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 15l7-7 7 7"></path></svg>
      </button>

    </div>
  );
};

export default Home;
