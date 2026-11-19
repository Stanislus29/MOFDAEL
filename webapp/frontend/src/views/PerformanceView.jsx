import EmptyState from "../components/EmptyState.jsx";
import { IconChart } from "../components/icons.jsx";

export default function PerformanceView() {
  return (
    <EmptyState
      icon={<IconChart size={40} />}
      title="Model performance"
      hint="Accuracy, latency, robustness and location-error comparisons
        across the nine benchmark methods — rendered from the
        dissertation's published metrics in the next build."
    />
  );
}
