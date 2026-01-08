import React from "react";
import { Routes, Route } from "react-router-dom";

// Import Components
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";

//import pages
import Home from "./components/home";
import MriUpload from "./components/upload_result";
import About from "./components/about";
import Tabs from "./components/Tab";
function App() {
  return (
    <div className="app-container flex flex-col min-h-screen">
      <Navbar />
      <main className="app-content flex-grow flex flex-col">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/upload" element={<MriUpload />} />
          <Route path="/about" element={<About />} />
          <Route path="/tab" element={<Tabs />} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}

export default App;
