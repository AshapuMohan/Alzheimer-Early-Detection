
import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
    ArrowRightIcon,
    ChartBarIcon,
    CloudArrowUpIcon,
    ShieldCheckIcon,
    SparklesIcon
} from '@heroicons/react/24/outline';

const LandingPage = () => {
    const [scrolled, setScrolled] = useState(false);

    useEffect(() => {
        const handleScroll = () => {
            const isScrolled = window.scrollY > 20;
            if (isScrolled !== scrolled) {
                setScrolled(isScrolled);
            }
        };

        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, [scrolled]);

    return (
        <div className="font-sans bg-white text-slate-800 min-h-screen selection:bg-teal-100 selection:text-teal-900">

            {/* Navbar Background for scroll state */}
            <div className={`fixed top-0 left-0 right-0 h-16 z-40 transition-all duration-300 ${scrolled ? 'bg-white/90 backdrop-blur-md shadow-sm border-b border-slate-100' : 'bg-transparent'}`}></div>

            {/* Hero Section */}
            <div className="relative pt-32 pb-20 sm:pt-40 sm:pb-24 lg:pb-32 overflow-hidden">
                {/* Subtle Background Shape - Teal/Emerald Gradient */}
                <div className="absolute top-0 right-0 -mr-20 -mt-20 w-[600px] h-[600px] bg-gradient-to-br from-teal-50 to-emerald-50 rounded-full blur-3xl opacity-60 pointer-events-none" />
                <div className="absolute bottom-0 left-0 -ml-20 -mb-20 w-[400px] h-[400px] bg-teal-50 rounded-full blur-3xl opacity-40 pointer-events-none" />

                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative text-center">
                    <div className="inline-flex items-center px-3 py-1 rounded-full bg-teal-50 border border-teal-100 text-sm font-medium text-teal-700 mb-8 animate-fade-in shadow-sm">
                        <SparklesIcon className="w-4 h-4 mr-2 text-teal-600" />
                        <span>AI-Powered Diagnostics</span>
                    </div>

                    <h1 className="text-5xl sm:text-7xl font-bold tracking-tight mb-6 font-['Outfit'] text-slate-900 leading-tight">
                        Precision Medicine for <br />
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-600 to-emerald-600">Alzheimer's Detection</span>
                    </h1>

                    <p className="mt-6 text-xl text-slate-600 max-w-2xl mx-auto mb-10 leading-relaxed font-['Inter']">
                        Detect early signs of cognitive decline with our clinically validated AI model. Fast, secure, and accessible from anywhere.
                    </p>

                    <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                        <Link
                            to="/upload"
                            className="inline-flex items-center justify-center px-8 py-4 text-lg font-medium text-white transition-all duration-200 bg-teal-600 rounded-full hover:bg-teal-700 hover:shadow-lg hover:shadow-teal-600/20 hover:-translate-y-0.5 active:translate-y-0 active:shadow-md"
                        >
                            <span className="mr-2">Start Analysis</span>
                            <ArrowRightIcon className="w-5 h-5" />
                        </Link>
                        <Link
                            to="/about"
                            className="inline-flex items-center justify-center px-8 py-4 text-lg font-medium text-slate-700 transition-all duration-200 bg-white border border-slate-200 rounded-full hover:bg-slate-50 hover:border-slate-300 shadow-sm"
                        >
                            Learn More
                        </Link>
                    </div>
                </div>
            </div>

            {/* Features Section (Material Cards) */}
            <div className="py-24 bg-slate-50">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="text-center mb-16">
                        <h2 className="text-teal-600 font-bold tracking-wide uppercase text-xs mb-3">Key Features</h2>
                        <h3 className="text-3xl sm:text-4xl font-bold text-slate-900 font-['Outfit']">
                            Why choose our platform?
                        </h3>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                        {[
                            {
                                icon: ChartBarIcon,
                                title: "High Accuracy",
                                desc: "Trained on thousands of verified MRI scans to identify subtle patterns invisible to the human eye."
                            },
                            {
                                icon: CloudArrowUpIcon,
                                title: "Instant Analysis",
                                desc: "Cloud-native architecture delivers comprehensive reports in seconds, enabling faster decision-making."
                            },
                            {
                                icon: ShieldCheckIcon,
                                title: "Secure & Private",
                                desc: "Enterprise-grade encryption ensures your medical data remains private and HIPAA-compliant."
                            }
                        ].map((feature, idx) => (
                            <div key={idx} className="bg-white rounded-3xl p-8 shadow-[0_2px_8px_rgba(0,0,0,0.04)] hover:shadow-[0_8px_24px_rgba(0,0,0,0.08)] transition-all duration-300 border border-slate-100 group">
                                <div className="w-14 h-14 bg-teal-50 rounded-2xl flex items-center justify-center mb-6 text-teal-600 group-hover:bg-teal-600 group-hover:text-white transition-colors duration-300">
                                    <feature.icon className="w-7 h-7" />
                                </div>
                                <h4 className="text-xl font-bold text-slate-900 mb-3 font-['Outfit']">{feature.title}</h4>
                                <p className="text-slate-600 leading-relaxed">
                                    {feature.desc}
                                </p>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* How It Works (Clean Steps) */}
            <div className="py-24 bg-white">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="text-center mb-20">
                        <h2 className="text-3xl sm:text-4xl font-bold text-slate-900 font-['Outfit']">
                            Three steps to insight
                        </h2>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-12 relative">
                        {/* Connecting Line (Desktop) */}
                        <div className="hidden md:block absolute top-[24px] left-[16%] right-[16%] h-[2px] bg-slate-100 -z-10"></div>

                        {[
                            { step: 1, title: 'Upload Scan', desc: 'Securely upload standard MRI file formats (JPG, PNG).' },
                            { step: 2, title: 'AI Processing', desc: 'Our model analyzes the image for specific biomarkers.' },
                            { step: 3, title: 'View Report', desc: 'Receive a detailed probability assessment instantly.' }
                        ].map((item, i) => (
                            <div key={i} className="flex flex-col items-center text-center group">
                                <div className="w-12 h-12 rounded-full bg-white border-2 border-teal-600 text-teal-600 flex items-center justify-center text-xl font-bold mb-6 shadow-sm group-hover:bg-teal-600 group-hover:text-white transition-colors duration-300">
                                    {item.step}
                                </div>
                                <h3 className="text-xl font-bold text-slate-900 mb-2 font-['Outfit']">{item.title}</h3>
                                <p className="text-slate-600 max-w-xs">{item.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Stats Section (Minimal) */}
            <div className="bg-slate-900 py-20 relative overflow-hidden">
                {/* Decorative gradients */}
                <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-r from-slate-900 to-slate-800 z-0"></div>
                <div className="absolute -top-24 -right-24 w-96 h-96 bg-teal-600/20 rounded-full blur-3xl z-0"></div>

                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-8 text-center text-white">
                        {[
                            { label: 'Model Accuracy', value: '~89.9%' },
                            { label: 'Processing Time', value: '< 2s' },
                            { label: 'Scans Analyzed', value: '10k+' },
                            { label: 'Availability', value: '24/7' },
                        ].map((stat, index) => (
                            <div key={index} className="flex flex-col">
                                <dd className="text-4xl sm:text-5xl font-bold font-['Outfit'] mb-2 text-transparent bg-clip-text bg-gradient-to-br from-white to-slate-300">{stat.value}</dd>
                                <dt className="text-teal-400 text-sm font-bold uppercase tracking-widest">{stat.label}</dt>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* CTA Section */}
            <div className="py-24 bg-white text-center">
                <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
                    <h2 className="text-3xl sm:text-4xl font-bold text-slate-900 mb-6 font-['Outfit']">
                        Ready to get started?
                    </h2>
                    <p className="text-lg text-slate-600 mb-10">
                        Join the thousands of users leveraging AI for early detection.
                    </p>
                    <Link
                        to="/upload"
                        className="inline-flex items-center justify-center px-10 py-4 text-lg font-medium text-white bg-teal-600 rounded-full hover:bg-teal-700 transition-colors shadow-lg shadow-teal-600/20"
                    >
                        Analyze MRI Scan
                    </Link>
                </div>
            </div>

        </div>
    );
};

export default LandingPage;
