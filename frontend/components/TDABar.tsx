interface Props {
  name: string;
  value: number;   // 0–1 importance
  rawValue?: number;
}

export default function TDABar({ name, value, rawValue }: Props) {
  const isH1 = name.includes("H1");
  const color = isH1 ? "bg-pink-500" : "bg-blue-500";
  const pct   = (value * 100).toFixed(0);

  return (
    <div className="mb-1.5">
      <div className="flex justify-between text-xs mb-0.5">
        <span className="text-slate-300 font-mono text-[11px]">{name}</span>
        <span className="text-slate-400 text-[11px]">
          {pct}%
          {rawValue !== undefined && (
            <span className="ml-1 text-slate-500">({rawValue.toFixed(3)})</span>
          )}
        </span>
      </div>
      <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${color} opacity-75 transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
