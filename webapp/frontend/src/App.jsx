import { useState } from "react";
import Sidebar from "./components/Sidebar.jsx";
import TopBar from "./components/TopBar.jsx";
import StudioView from "./views/StudioView.jsx";
import MonitorView from "./views/MonitorView.jsx";
import LabView from "./views/LabView.jsx";
import PerformanceView from "./views/PerformanceView.jsx";

const TITLES = {
  studio: "Network Studio",
  monitor: "Line Monitor",
  lab: "Model Lab",
  performance: "Performance",
};

export default function App() {
  const [view, setView] = useState("lab");

  return (
    <div className="shell">
      <Sidebar active={view} onNavigate={setView} />
      <main className="main">
        <TopBar title={TITLES[view]} />
        <div className="content">
          {view === "studio" && <StudioView onNavigate={setView} />}
          {view === "monitor" && <MonitorView />}
          {view === "lab" && <LabView />}
          {view === "performance" && <PerformanceView />}
        </div>
      </main>
    </div>
  );
}
