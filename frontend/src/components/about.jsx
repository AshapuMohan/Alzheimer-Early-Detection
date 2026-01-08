import { useState } from 'react'
import {  Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react'

// Simple SVG icons for the pillars. In a real app, these might be more detailed.
const TechnologyIcon = () => (
  <svg className="w-12 h-12 mx-auto text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.293 2.293a1 1 0 010 1.414L11 15H9v-2l6.293-6.293a1 1 0 011.414 0z" /></svg>
);
const ImpactIcon = () => (
  <svg className="w-12 h-12 mx-auto text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
);
const MissionIcon = () => (
  <svg className="w-12 h-12 mx-auto text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
);


const About = () => {
  const [open, setOpen] = useState(false)
  const pillars = [
    {
      icon: <TechnologyIcon />,
      title: "Our Technology",
      content: "This tool leverages a sophisticated deep learning model, trained on vast datasets of MRI scans to recognize subtle patterns associated with different stages of dementia."
    },
    {
      icon: <ImpactIcon />,
      title: "The Impact",
      content: "By providing early-stage analysis, we empower individuals and healthcare providers with information that can lead to better planning and management of cognitive health."
    },
    {
      icon: <MissionIcon />,
      title: "Our Mission",
      content: "To make cutting-edge diagnostic technology accessible and understandable, offering a clear first step in the journey of brain health awareness."
    }
  ];

  return (
    <div className="bg-white text-gray-700 font-sans">
      {/* Hero Section */}
      <div className="text-center py-24 md:py-32 px-6 bg-gray-50">
        <h1 className="text-4xl md:text-5xl font-extrabold text-gray-900 tracking-tight">
          Pioneering Clarity in Cognitive Health
        </h1>
        <p className="mt-6 text-lg md:text-xl text-gray-600 max-w-3xl mx-auto">
          We use the power of artificial intelligence to shed light on one of the most complex challenges in modern medicine. This project is a demonstration of how technology can serve humanity by providing early insights into Alzheimer's disease.
        </p>
      </div>

      {/* Pillars Section */}
      <div className="py-24 px-6">
        <div className="max-w-6xl mx-auto grid md:grid-cols-3 gap-12 md:gap-16 text-center">
          {pillars.map((pillar, index) => (
            <div key={index}>
              {pillar.icon}
              <h2 className="mt-6 text-2xl font-bold text-gray-900">{pillar.title}</h2>
              <p className="mt-4 text-base text-gray-600">{pillar.content}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Disclaimer Section */}
      <div className="py-20 px-6 text-center">
        <h3 className="text-xl font-semibold text-gray-800">For Educational Purposes Only</h3>
        <p className="max-w-2xl mx-auto text-gray-600 mt-4">
          This tool is designed for educational and research purposes and is not a substitute for professional medical diagnosis. Always consult with a qualified healthcare provider for any health concerns.
        </p>
      </div>
      
      {/* Team Section */}
      <div className=' text-center pb-10'>
        <button
          onClick={() => setOpen(true)}
          className="rounded-md bg-gray-950/5 px-2.5 py-1.5 text-sm font-semibold text-gray-900 hover:bg-gray-950/10"
        >
          Meet my Developer
        </button>
        <Dialog open={open} onClose={setOpen} className="relative z-10">
          <DialogBackdrop
            transition
            className="fixed inset-0 bg-gray-500/75 transition-opacity data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in"
          />

          <div className="fixed inset-0 z-10 w-screen overflow-y-auto">
            <div className="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0">
              <DialogPanel
                transition
                className="relative transform overflow-hidden rounded-lg bg-white text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in sm:my-8 sm:w-full sm:max-w-lg data-closed:sm:translate-y-0 data-closed:sm:scale-95"
              >
                <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                  <div className="sm:flex sm:items-start">
                    <div className="mx-auto flex size-12 shrink-0 items-center justify-center rounded-full bg-red-100 sm:mx-0 sm:size-10">
                     <img src='/Mohan.jpg' alt='Developer Photo' className='rounded-full'/>
                    </div>
                    <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                      <DialogTitle as="h3" className="text-base font-semibold text-gray-900">
                        Ashapu Mohan
                      </DialogTitle>
                      <div className="mt-2">
                        <p className="text-sm text-gray-500">
                          A passionate developer with a keen interest in leveraging technology to solve real-world problems. With a background in computer science and a love for innovation, he is dedicated to creating impactful solutions that make a difference.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="bg-gray-100 px-4 py-3 sm:flex sm:flex-row-reverse sm:px-6">
                  <button
                    type="button"
                    data-autofocus
                    onClick={() => setOpen(false)}
                    className="mt-3 inline-flex w-full justify-center rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-xs inset-ring inset-ring-gray-300 hover:bg-gray-50 sm:mt-0 sm:w-auto"
                  >
                    Cancel
                  </button>
                </div>
              </DialogPanel>
            </div>
          </div>
        </Dialog>
      </div>

    </div>
  );
};

export default About;
