import EmptyState from "../components/EmptyState.jsx";
import { IconMonitor } from "../components/icons.jsx";

export default function MonitorView() {
  return (
    <EmptyState
      icon={<IconMonitor size={40} />}
      title="Live line monitoring"
      hint="Once a network exists in the Studio, this view renders it as a
        schematic — energized segments, relay trip flags and fault heat
        zones where the models agree a fault is present."
    />
  );
}
