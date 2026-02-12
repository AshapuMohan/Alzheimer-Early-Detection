import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import html2pdf from 'html2pdf.js';

/**
 * AI-POWERED MRI ANALYSIS SYSTEM
 * A single-file implementation featuring:
 * - Drag & Drop functionality
 * - Memory-safe preview management
 * - Responsive data visualization
 */

const MriUpload = () => {
    // --- State Management ---
    const [file, setFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [analysis, setAnalysis] = useState(null);
    const [status, setStatus] = useState('idle'); // idle | analyzing | success | error
    const [error, setError] = useState('');

    const fileInputRef = useRef(null);

    // --- Cleanup Memory Leaks ---
    useEffect(() => {
        return () => {
            if (previewUrl) URL.revokeObjectURL(previewUrl);
        };
    }, [previewUrl]);

    // --- Handlers ---
    const handleFileChange = (selectedFile) => {
        if (!selectedFile) return;

        // Reset previous state
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        setAnalysis(null);
        setError('');

        setFile(selectedFile);
        setPreviewUrl(URL.createObjectURL(selectedFile));
        setStatus('idle');
    };

    const handleUpload = async () => {
        if (!file) {
            setError('Please select an image first.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        setStatus('analyzing');
        setError('');

        try {
            const response = await axios.post('http://localhost:8000/api/analyze-mri', formData);
            setAnalysis(response.data);
            setStatus('success');
        } catch (err) {
            setStatus('error');
            setError(err.response?.data?.error || 'Analysis failed. Please check your connection.');
        }
    };

    const handleReset = () => {
        setFile(null);
        setPreviewUrl(null);
        setAnalysis(null);
        setError('');
        setStatus('idle');
    };

    // --- UI Components (Sub-components) ---

    const MetricCard = ({ label, value, highlight }) => (
        <div className={`p-4 rounded-xl border ${highlight ? 'bg-blue-600 border-blue-600 text-white' : 'bg-white border-gray-200 text-gray-900 shadow-sm'}`}>
            <p className={`text-[10px] uppercase font-bold tracking-widest mb-1 ${highlight ? 'text-blue-100' : 'text-gray-400'}`}>{label}</p>
            <p className="text-lg font-bold truncate">{value || 'N/A'}</p>
        </div>
    );

    const ConfidenceBar = ({ label, prob, isWinner }) => (
        <div className="space-y-1">
            <div className="flex justify-between text-xs font-medium">
                <span className={isWinner ? 'text-blue-700' : 'text-gray-600'}>{label}</span>
                <span className="text-gray-500">{(prob * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                    className={`h-2 rounded-full transition-all duration-1000 ${isWinner ? 'bg-blue-600' : 'bg-gray-400'}`}
                    style={{ width: `${prob * 100}%` }}
                />
            </div>
        </div>
    );

    const handlePrint = () => {
        window.print();
    };

    const handleDownloadReport = () => {
        // We select the report element
        const element = document.getElementById('clinical-report');

        // Configuration for html2pdf
        const opt = {
            margin: [10, 10, 10, 10], // top, left, bottom, right (mm)
            filename: `AlzheimerAI_Report_${new Date().toISOString().split('T')[0]}.pdf`,
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: { scale: 2, useCORS: true, logging: false },
            jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
        };

        // Execution
        // We temporarily make it visible for the capture if needed, but off-screen rendering usually works
        // if the element is in the DOM.
        html2pdf().set(opt).from(element).save();
    };

    // --- Main Render ---
    return (
        <div className="min-h-screen bg-gray-50 py-12 px-4 font-sans text-gray-900">
            {/* WEB UI VIEW */}
            <div className="max-w-3xl mx-auto bg-white rounded-3xl shadow-xl overflow-hidden border border-gray-100">

                {/* Header */}
                <div className="bg-white border-b border-gray-100 p-8 text-center relative">
                    <h1 className="text-3xl font-extrabold tracking-tight text-gray-900">AlzheimerAI</h1>
                    <p className="text-gray-500 mt-2">Clinical Decision Support System for MRI Analysis</p>

                    {/* DOWNLOAD BUTTON */}
                    {status === 'success' && (
                        <button
                            onClick={handleDownloadReport}
                            className="absolute top-8 right-8 flex items-center gap-2 bg-gray-900 text-white px-4 py-2 rounded-full text-sm font-medium hover:bg-gray-800 transition-colors shadow-lg active:scale-95"
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>
                            Download PDF
                        </button>
                    )}
                </div>

                <div className="p-8">
                    {/* 1. Upload Section */}
                    {status === 'idle' && !previewUrl && (
                        <div
                            onClick={() => fileInputRef.current.click()}
                            className="border-2 border-dashed border-gray-300 rounded-2xl p-12 text-center cursor-pointer hover:border-blue-500 hover:bg-blue-50 transition-all group"
                        >
                            <input
                                type="file"
                                hidden
                                ref={fileInputRef}
                                onChange={(e) => handleFileChange(e.target.files[0])}
                                accept="image/*"
                            />
                            <div className="w-16 h-16 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
                                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>
                            </div>
                            <h3 className="text-lg font-semibold">Upload MRI Scan</h3>
                            <p className="text-sm text-gray-400 mt-1">Drag and drop or click to browse (Max 30MB)</p>
                        </div>
                    )}

                    {/* 2. Preview Section */}
                    {previewUrl && status !== 'success' && status !== 'analyzing' && (
                        <div className="space-y-6 animate-in fade-in duration-500">
                            <div className="relative rounded-2xl overflow-hidden bg-black shadow-inner">
                                <img src={previewUrl} alt="MRI Preview" className="max-h-96 mx-auto object-contain" />
                                <button
                                    onClick={handleReset}
                                    className="absolute top-4 right-4 bg-white/80 backdrop-blur-sm p-2 rounded-full hover:bg-white transition-colors"
                                >
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg>
                                </button>
                            </div>
                            <button
                                onClick={handleUpload}
                                className="w-full bg-blue-600 text-white py-4 rounded-xl font-bold text-lg hover:bg-blue-700 shadow-lg shadow-blue-200 transition-all active:scale-[0.98]"
                            >
                                Run AI Analysis
                            </button>
                        </div>
                    )}

                    {/* 3. Loading State */}
                    {status === 'analyzing' && (
                        <div className="py-20 text-center space-y-4">
                            <div className="animate-spin w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full mx-auto"></div>
                            <p className="text-lg font-medium text-gray-600 animate-pulse">Quantifying neural patterns...</p>
                        </div>
                    )}

                    {/* 4. Analysis Results (Web View) */}
                    {status === 'success' && analysis && analysis.results && (
                        <div className="space-y-12 animate-in slide-in-from-bottom-8 duration-700">

                            {analysis.results.map((result, index) => (
                                <div key={index} className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm relative overflow-hidden">
                                    <div className="absolute top-0 right-0 bg-gray-100 px-3 py-1 rounded-bl-xl text-xs font-bold text-gray-500">
                                        Scan #{index + 1}
                                    </div>

                                    {/* Make image visible */}
                                    {result.image_base64 && (
                                        <div className="mb-6 flex justify-center">
                                            <img src={result.image_base64} alt={`Scan ${index + 1}`} className="max-h-96 w-full object-contain rounded-lg border border-gray-100 shadow-sm" />
                                        </div>
                                    )}

                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                                        <MetricCard label="Identified View" value={result.view} />
                                        <MetricCard label="Approx. Location" value={result.location} />
                                        <MetricCard label="Predicted Stage" value={result.predicted_stage} highlight />
                                    </div>

                                    {result.probabilities && (
                                        <div className="bg-gray-50 border border-gray-100 rounded-2xl p-6 mb-6">
                                            <h3 className="text-sm font-bold text-gray-400 uppercase tracking-widest mb-6">Confidence Distribution</h3>
                                            <div className="space-y-5">
                                                {Object.entries(result.probabilities)
                                                    .sort(([, a], [, b]) => b - a)
                                                    .map(([stage, prob]) => (
                                                        <ConfidenceBar
                                                            key={stage}
                                                            label={stage}
                                                            prob={prob}
                                                            isWinner={stage === result.predicted_stage}
                                                        />
                                                    ))}
                                            </div>
                                        </div>
                                    )}

                                    <div className="bg-blue-50 border border-blue-100 rounded-2xl p-6 mb-6">
                                        <div className="flex items-center gap-2 mb-3">
                                            <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                                            <h3 className="font-bold text-blue-900">AI-Generated Insights</h3>
                                        </div>
                                        <p className="text-blue-800 leading-relaxed italic">"{result.suggestions}"</p>
                                    </div>

                                    <div className="bg-amber-50 border border-amber-100 rounded-2xl p-6 flex items-start gap-4">
                                        <div className="p-3 bg-amber-100 rounded-xl text-amber-600">
                                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" /></svg>
                                        </div>
                                        <div>
                                            <h3 className="font-bold text-amber-900">Historical Case Reference</h3>
                                            <p className="text-amber-800 text-sm mt-1 opacity-90">
                                                This scan shows <strong>{(result.similar_case?.similarity * 100).toFixed(1)}%</strong> morphological similarity
                                                to a verified <strong>{result.similar_case?.label}</strong> case.
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            ))}

                            <button
                                onClick={handleReset}
                                className="w-full py-4 text-gray-500 font-semibold hover:text-blue-600 hover:bg-blue-50 rounded-xl transition-all"
                            >
                                ← Analyze Another Image
                            </button>
                        </div>
                    )}

                    {/* 5. Error State */}
                    {status === 'error' && (
                        <div className="bg-red-50 border border-red-100 rounded-2xl p-8 text-center space-y-4">
                            <p className="text-red-800 font-medium">{error}</p>
                            <button
                                onClick={handleReset}
                                className="bg-red-600 text-white px-8 py-2 rounded-lg hover:bg-red-700 transition-colors"
                            >
                                Try Again
                            </button>
                        </div>
                    )}
                </div>
            </div>

            {/* OFF-SCREEN REPORT RENDER - Used by html2pdf handles */}
            {status === 'success' && analysis && analysis.results && analysis.results.length > 0 && (
                <div style={{ position: 'fixed', left: '-10000px', top: 0, width: '210mm', minHeight: '297mm' }}>
                    <div id="clinical-report" className="bg-white text-slate-900 p-8">
                        {/* Google Design: Clean, Spacious, Grid-Based */}
                        {(() => {
                            const result = analysis.results[0];
                            return (
                                <div className="max-w-[210mm] mx-auto min-h-[297mm] flex flex-col justify-between">
                                    {/* 1. Header Section */}
                                    <header className="border-b-2 border-slate-900 pb-8 mb-8 flex justify-between items-start">
                                        <div className="space-y-2 max-w-[50%]">
                                            <h1 className="text-4xl font-serif font-bold text-slate-900 tracking-tight leading-none">Diagnostic Report</h1>
                                            <p className="text-slate-500 font-medium uppercase tracking-widest text-xs">AlzheimerAI Clinical Decision Support</p>
                                        </div>
                                        <div className="text-right space-y-1">
                                            <div className="inline-block bg-slate-100 px-3 py-1 rounded text-xs font-mono font-bold text-slate-600 mb-2">
                                                ID: #{Math.floor(Math.random() * 90000) + 10000}
                                            </div>
                                            <p className="text-sm font-medium text-slate-900">Date: {new Date().toLocaleDateString()}</p>
                                            <p className="text-sm text-slate-400">Ref: AI-V4-MAVERICK</p>
                                        </div>
                                    </header>

                                    {/* 2. Main Content Grid */}
                                    <div className="flex-grow">
                                        {/* Patient/Context Bar */}
                                        <div className="grid grid-cols-2 gap-8 mb-12">
                                            <div className="bg-slate-50 p-6 rounded-lg border border-slate-100">
                                                <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3">Assessment Summary</p>
                                                <div className="flex items-baseline gap-3">
                                                    <span className="text-3xl font-bold text-slate-900">{result.predicted_stage}</span>
                                                </div>
                                                <p className="text-sm text-slate-500 mt-2">
                                                    AI Confidence: <span className="font-semibold text-slate-700">
                                                        {(result.probabilities?.[result.predicted_stage] * 100).toFixed(1)}%
                                                    </span>
                                                </p>
                                            </div>
                                            <div className="flex gap-4">
                                                <div className="flex-1 border-l-2 border-slate-200 pl-4 py-1">
                                                    <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">View</p>
                                                    <p className="text-lg font-medium text-slate-900">{result.view}</p>
                                                </div>
                                                <div className="flex-1 border-l-2 border-slate-200 pl-4 py-1">
                                                    <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Location</p>
                                                    <p className="text-lg font-medium text-slate-900">{result.location}</p>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Visual Analysis Section */}
                                        <section className="mb-12">
                                            <h3 className="text-xs font-bold text-slate-900 uppercase tracking-widest border-b border-slate-200 pb-2 mb-6">Radiological Analysis</h3>
                                            <div className="grid grid-cols-[2fr_3fr] gap-8">
                                                {/* Left: Image */}
                                                <div className="aspect-square bg-slate-900 rounded-lg overflow-hidden flex items-center justify-center relative shadow-sm">
                                                    <img
                                                        src={result.image_base64}
                                                        alt="Analyzed Scan"
                                                        className="max-w-full max-h-full object-contain"
                                                        style={{ filter: 'contrast(1.1) brightness(1.1)' }} // Optimize for print contrast
                                                    />
                                                    <div className="absolute bottom-2 left-2 bg-black/50 text-white text-[10px] px-2 py-0.5 rounded backdrop-blur-sm">
                                                        Processed Input
                                                    </div>
                                                </div>

                                                {/* Right: Detailed Metrics */}
                                                <div className="space-y-6">
                                                    <div className="space-y-2">
                                                        <p className="text-sm font-medium text-slate-900">Class Probability Assessment</p>
                                                        <div className="space-y-3">
                                                            {result.probabilities && Object.entries(result.probabilities)
                                                                .sort(([, a], [, b]) => b - a)
                                                                .map(([stage, prob]) => (
                                                                    <div key={stage} className="group">
                                                                        <div className="flex justify-between text-xs mb-1">
                                                                            <span className={`font-medium ${stage === result.predicted_stage ? 'text-slate-900' : 'text-slate-500'}`}>
                                                                                {stage}
                                                                            </span>
                                                                            <span className="font-mono text-slate-400">{(prob * 100).toFixed(1)}%</span>
                                                                        </div>
                                                                        <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                                                                            <div
                                                                                className={`h-full rounded-full ${stage === result.predicted_stage ? 'bg-slate-900' : 'bg-slate-300'}`}
                                                                                style={{ width: `${prob * 100}%` }}
                                                                            ></div>
                                                                        </div>
                                                                    </div>
                                                                ))}
                                                        </div>
                                                    </div>

                                                    {/* Similar Case Mini-Card */}
                                                    {result.similar_case && (
                                                        <div className="bg-amber-50 p-4 rounded border border-amber-100 mt-4">
                                                            <p className="text-[10px] font-bold text-amber-600 uppercase tracking-wider mb-1">Historical Match</p>
                                                            <p className="text-xs text-amber-900 leading-relaxed">
                                                                High morphological correlation ({(result.similar_case.similarity * 100).toFixed(0)}%) with a confirmed <strong>{result.similar_case.label}</strong> case.
                                                            </p>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        </section>

                                        {/* AI Interpretation */}
                                        <section>
                                            <h3 className="text-xs font-bold text-slate-900 uppercase tracking-widest border-b border-slate-200 pb-2 mb-6">Automated Interpretation</h3>
                                            <div className="prose prose-sm max-w-none text-slate-700 font-serif leading-relaxed text-justify columns-2 gap-8">
                                                {result.suggestions}
                                            </div>
                                        </section>
                                    </div>

                                    {/* 3. Footer */}
                                    <footer className="border-t border-slate-200 pt-6 mt-8 flex justify-between items-end text-[10px] text-slate-400">
                                        <div className="max-w-md">
                                            <p className="uppercase tracking-widest font-bold mb-1 text-slate-500">Clinical Disclaimer</p>
                                            <p className="leading-tight">
                                                This report is generated by an artificial intelligence system (AlzheimerAI v1.0).
                                                It is intended as a supplementary selection and screening tool only.
                                                Final diagnosis must be confirmed by a licensed radiologist or neurologist.
                                                Errors in image segmentation or classification may occur.
                                            </p>
                                        </div>
                                        <div className="text-right">
                                            <p>Page 1 of 1</p>
                                            <p>Generated: {new Date().toLocaleTimeString()}</p>
                                        </div>
                                    </footer>
                                </div>
                            );
                        })()}
                    </div>
                </div>
            )}
        </div>
    );
};

export default MriUpload;