import { useEffect, useState } from "react";
import { api } from "../api/client.js";

export default function TopBar({ title, meta }) {
  const [backend, setBackend] = useState("checking");

  useEffect(() => {
    let live = true;
    api
      .health()
      .then(() => live && setBackend("online"))
      .catch(() => live && setBackend("offline"));
    const t = setInterval(() => {
      api
        .health()
        .then(() => live && setBackend("online"))
        .catch(() => live && setBackend("offline"));
    }, 15000);
    return () => {
      live = false;
      clearInterval(t);
    };
  }, []);

  return (
    <header className="topbar">
      <div className="topbar-title">{title}</div>
      <div className="topbar-meta">
        {meta}
        <span
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
          }}
          title={
            backend === "offline"
              ? "Backend not reachable on :8790 — start it with webapp/run_backend.sh (or ./run_platform.sh for both)."
              : undefined
          }
        >
          <span
            style={{
              width: 7,
              height: 7,
              borderRadius: "50%",
              background:
                backend === "online"
                  ? "var(--deep-blue)"
                  : backend === "checking"
                    ? "var(--hairline)"
                    : "#b04a3a",
            }}
          />
          API {backend}
        </span>
      </div>
    </header>
  );
}
