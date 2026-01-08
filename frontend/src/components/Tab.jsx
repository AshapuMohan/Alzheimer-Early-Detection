import { useState } from "react";
import "./tabs.css";
import Modal from 'react-modal';
Modal.setAppElement('#root')
export default function Tabs() {
  const [active, setActive] = useState("tab1");
  const [closeTab, setClosedTab] = useState(false);
  const [isOpen, setIsOpen] = useState(false)
  const handleCloseTab = () => {
    setClosedTab(true);
  }
  const handleOpenTab = () => {
    setClosedTab(false);
  }
  const [modelOpen,setModelOpen]=useState(false)
  return (
    <div className="tab-container">
      <div className="tabgroup" style={{ display: closeTab ? "none" : "block" }}>
        <div className="tabs">
          {["tab1", "tab2", "tab3"].map(tab => (
            <button
              key={tab}
              className={`tab ${active === tab ? "active" : ""}`}
              onClick={() => setActive(tab)}
            >
              {tab.toUpperCase()}
            </button>
          ))}
        </div>
        <div className="content">
          {active === "tab1" && <p>Tab 1 content</p>}
          {active === "tab2" && <p>Tab 2 content</p>}
          {active === "tab3" && <p>Tab 3 content</p>}
        </div>
      </div>
      <button className="close" onClick={handleCloseTab}>Close Tab</button>
      <button className="open" onClick={handleOpenTab}>Open Tab</button>
      <div style={{ display: "flex", height: "500px", border: "2px solid gray", width: "500px", justifyContent: "center", alignItems: "center", marginTop: "10px", borderRadius: "10px" }}>
        <button onClick={()=>{setIsOpen(true)}} className="flex bg-slate-500 text-stone-950 rounded-full px-2 py-1">Open Model</button>
        <Modal isOpen={isOpen} contentLabel="Example Modal" className="w-60 h-40 bg-white p-4 rounded-lg shadow-lg flex items-center justify-center flex-col" overlayClassName="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
          <div className="flex items-center w-full justify-center">
            <img src="/Mohan.jpg" alt="my photo" className="w-7 h-7 rounded-full"/>
            <h2 className="text-lg font-semibold mb-4">Modal Title</h2>
            <button onClick={()=>{setIsOpen(false)}} className="ml-auto text-gray-500 hover:text-gray-700 focus:outline-none">
              &times;
            </button>
          </div>
        </Modal>
      </div>
    </div>

  );
}
