import { NAV_ITEMS } from "./nav.js";

export default function Sidebar({ active, onNavigate }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-name">MOFDAEL</div>
        <div className="brand-sub">Line Fault Detection Platform</div>
      </div>

      <nav className="nav">
        {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            className={`nav-item${active === id ? " active" : ""}`}
            onClick={() => onNavigate(id)}
          >
            <Icon />
            {label}
          </button>
        ))}
      </nav>

      <div className="nav-foot">
        8 ML architectures + distance relay
        <br />
        161 kV / 120 km benchmark
      </div>
    </aside>
  );
}
