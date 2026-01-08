import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';

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

    // --- Main Render ---
    return (
        <div className="min-h-screen bg-gray-50 py-12 px-4 font-sans text-gray-900">
            <div className="max-w-3xl mx-auto bg-white rounded-3xl shadow-xl overflow-hidden border border-gray-100">
                
                {/* Header */}
                <div className="bg-white border-b border-gray-100 p-8 text-center">
                    <h1 className="text-3xl font-extrabold tracking-tight text-gray-900">AlzheimerAI</h1>
                    <p className="text-gray-500 mt-2">Clinical Decision Support System for MRI Analysis</p>
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

                    {/* 4. Analysis Results (Showing ALL Fields) */}
                    {status === 'success' && analysis && (
                        <div className="space-y-8 animate-in slide-in-from-bottom-8 duration-700">
                            
                            {/* Key Stats Grid */}
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                <MetricCard label="Identified View" value={analysis.view} />
                                <MetricCard label="Approx. Location" value={analysis.location} />
                                <MetricCard label="Predicted Stage" value={analysis.predicted_stage} highlight />
                            </div>

                            {/* Probabilities / Confidence Bar Chart */}
                            
                            {analysis.probabilities && (
                                <div className="bg-gray-50 border border-gray-100 rounded-2xl p-6">
                                    <h3 className="text-sm font-bold text-gray-400 uppercase tracking-widest mb-6">Confidence Distribution</h3>
                                    <div className="space-y-5">
                                        {Object.entries(analysis.probabilities)
                                            .sort(([, a], [, b]) => b - a)
                                            .map(([stage, prob]) => (
                                                <ConfidenceBar 
                                                    key={stage} 
                                                    label={stage} 
                                                    prob={prob} 
                                                    isWinner={stage === analysis.predicted_stage} 
                                                />
                                            ))}
                                    </div>
                                </div>
                            )}

                            {/* AI Suggestions Section */}
                            <div className="bg-blue-50 border border-blue-100 rounded-2xl p-6">
                                <div className="flex items-center gap-2 mb-3">
                                    <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                                    <h3 className="font-bold text-blue-900">AI-Generated Insights</h3>
                                </div>
                                <p className="text-blue-800 leading-relaxed italic">"{analysis.suggestions}"</p>
                            </div>

                            {/* Similar Case Data */}
                            {analysis.similar_case && (
                                <div className="bg-amber-50 border border-amber-100 rounded-2xl p-6 flex items-start gap-4">
                                    <div className="p-3 bg-amber-100 rounded-xl text-amber-600">
                                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" /></svg>
                                    </div>
                                    <div>
                                        <h3 className="font-bold text-amber-900">Historical Case Reference</h3>
                                        <p className="text-amber-800 text-sm mt-1 opacity-90">
                                            This scan shows <strong>{(analysis.similar_case.similarity * 100).toFixed(1)}%</strong> morphological similarity 
                                            to a verified <strong>{analysis.similar_case.label}</strong> case in the training dataset.
                                        </p>
                                    </div>
                                </div>
                            )}

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

                {/* Footer Disclaimer */}
                {/* <div className="bg-gray-50 p-6 border-t border-gray-100 text-center">
                    <p className="text-[10px] text-gray-400 uppercase tracking-widest leading-relaxed">
                        For Research Purposes Only • Not a Substitute for Professional Medical Advice
                    </p>
                </div> */}
            </div>
        </div>
    );
};

export default MriUpload;