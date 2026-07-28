import { CLASS_COLOR_TW } from "../lib/config";

interface Props {
  label: string;
  value: number;        // 0–1
  highlighted?: boolean;
  color?: string;
}

export default function ConfidenceBar({ label, value, highlighted = false, color }: Props) {
  const pct = (value * 100).toFixed(1);
  const barColor = color ?? CLASS_COLOR_TW[label] ?? "bg-teal-500";

  return (
    <div className={`mb-2 ${highlighted ? "ring-1 ring-teal-400 rounded-lg p-2 -mx-2" : ""}`}>
      <div className="flex justify-between text-xs mb-1">
        <span className={highlighted ? "text-teal-300 font-semibold" : "text-slate-300"}>
          {highlighted ? "▶ " : ""}{label}
        </span>
        <span className={highlighted ? "text-teal-300 font-semibold" : "text-slate-400"}>
          {pct}%
        </span>
      </div>
      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full conf-bar ${highlighted ? "bg-teal-400" : barColor} opacity-80`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
