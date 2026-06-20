import { ReactNode, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const AXIS = { fill: "#94a3b8", fontSize: 11 };
const GRID = "#94a3b820";
const TOOLTIP_STYLE = {
  background: "#0f172a",
  border: "1px solid #334155",
  borderRadius: "12px",
  boxShadow: "0 18px 40px rgba(15, 23, 42, .2)",
  color: "#f8fafc",
  fontSize: "12px",
};

export function Card({
  title,
  eyebrow,
  right,
  children,
  className = "",
  noPadding = false,
}: {
  title?: string;
  eyebrow?: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  noPadding?: boolean;
}) {
  return (
    <section className={`card ${noPadding ? "!p-0" : ""} ${className}`}>
      {(title || eyebrow || right) && (
        <div
          className={`flex items-start justify-between gap-4 ${
            noPadding ? "px-5 pt-5 sm:px-6 sm:pt-6" : "mb-5"
          }`}
        >
          <div>
            {eyebrow && <p className="eyebrow">{eyebrow}</p>}
            {title && (
              <h3
                className={`${
                  eyebrow ? "mt-1.5" : ""
                } text-base font-semibold tracking-tight text-slate-950 dark:text-white`}
              >
                {title}
              </h3>
            )}
          </div>
          {right}
        </div>
      )}
      {children}
    </section>
  );
}

export function Metric({
  icon,
  value,
  label,
  tone = "blue",
}: {
  icon: ReactNode;
  value: ReactNode;
  label: string;
  tone?: "blue" | "teal" | "amber" | "violet";
}) {
  const tones = {
    blue: "metric-blue",
    teal: "metric-teal",
    amber: "metric-amber",
    violet: "metric-violet",
  };
  return (
    <div className="metric-card">
      <div className={`metric-icon ${tones[tone]}`}>{icon}</div>
      <div className="min-w-0">
        <div className="truncate text-2xl font-semibold tracking-[-0.03em] text-slate-950 dark:text-white">
          {value}
        </div>
        <div className="mt-0.5 truncate text-xs font-medium text-slate-500 dark:text-slate-400">
          {label}
        </div>
      </div>
    </div>
  );
}

export function Stepper({ steps }: { steps: string[] }) {
  return (
    <div className="overflow-x-auto pb-1">
      <div className="flex min-w-[620px] items-start">
        {steps.map((step, index) => (
          <div className="relative flex flex-1 flex-col items-center" key={step}>
            {index < steps.length - 1 && (
              <div className="absolute left-1/2 top-4 h-0.5 w-full bg-teal-500/50" />
            )}
            <div className="relative z-10 grid h-8 w-8 place-items-center rounded-full border-4 border-white bg-teal-500 text-xs font-bold text-white shadow-sm dark:border-slate-900">
              ✓
            </div>
            <span className="mt-2 text-[11px] font-semibold text-slate-500 dark:text-slate-400">
              {step}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function Tabs({
  tabs,
  children,
}: {
  tabs: string[];
  children: (active: string) => ReactNode;
}) {
  const [active, setActive] = useState(tabs[0]);
  return (
    <div>
      <div className="border-b border-slate-200 px-4 pt-2 dark:border-slate-800 sm:px-6">
        <div className="flex gap-1 overflow-x-auto">
          {tabs.map((tab) => (
            <button
              className={`tab-button ${active === tab ? "tab-button-active" : ""}`}
              key={tab}
              onClick={() => setActive(tab)}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>
      {children(active)}
    </div>
  );
}

const VIRIDIS = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
];
const DIVERGING = [
  [37, 99, 235],
  [226, 232, 240],
  [225, 29, 72],
];

function interpolate(stops: number[][], value: number): string {
  const bounded = Math.max(0, Math.min(1, value));
  const position = bounded * (stops.length - 1);
  const index = Math.floor(position);
  const fraction = position - index;
  const start = stops[index];
  const end = stops[Math.min(index + 1, stops.length - 1)];
  const color = start.map((channel, channelIndex) =>
    Math.round(channel + (end[channelIndex] - channel) * fraction),
  );
  return `rgb(${color[0]},${color[1]},${color[2]})`;
}

export function Heatmap({
  matrix,
  labels,
}: {
  matrix: number[][];
  labels: string[];
}) {
  const size = matrix.length;
  if (!size) return null;
  const values = matrix.flat();
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const diverging = minimum < 0;
  const absoluteMaximum = Math.max(Math.abs(minimum), Math.abs(maximum)) || 1;
  const span = maximum - minimum || 1;
  const cell = Math.max(5, Math.min(28, Math.floor(620 / size)));
  const showLabels = size <= 28;
  const padding = showLabels ? 112 : 10;
  const plotSize = size * cell;
  const colorFor = (value: number) =>
    diverging
      ? interpolate(DIVERGING, (value + absoluteMaximum) / (2 * absoluteMaximum))
      : interpolate(VIRIDIS, (value - minimum) / span);

  return (
    <div>
      <div className="heatmap-shell">
        <svg
          aria-label={`${size} by ${size} matrix heatmap`}
          className="min-w-[560px]"
          shapeRendering="crispEdges"
          style={{ maxHeight: 620 }}
          viewBox={`0 0 ${plotSize + padding} ${plotSize + padding}`}
          width="100%"
        >
          {matrix.map((row, rowIndex) =>
            row.map((value, columnIndex) => (
              <rect
                fill={colorFor(value)}
                height={cell}
                key={`${rowIndex}-${columnIndex}`}
                width={cell}
                x={padding + columnIndex * cell}
                y={rowIndex * cell}
              >
                <title>{`${labels[rowIndex]} × ${labels[columnIndex]}: ${value.toFixed(
                  6,
                )}`}</title>
              </rect>
            )),
          )}
          {showLabels &&
            labels.map((label, index) => (
              <text
                fill="#64748b"
                fontSize={9}
                key={`y-${label}-${index}`}
                textAnchor="end"
                x={padding - 8}
                y={index * cell + cell / 2 + 3}
              >
                {label.length > 17 ? `${label.slice(0, 16)}…` : label}
              </text>
            ))}
          {showLabels &&
            labels.map((label, index) => (
              <text
                fill="#64748b"
                fontSize={9}
                key={`x-${label}-${index}`}
                textAnchor="end"
                transform={`rotate(-58 ${padding + index * cell + cell / 2} ${
                  plotSize + 16
                })`}
                x={padding + index * cell + cell / 2}
                y={plotSize + 16}
              >
                {label.length > 15 ? `${label.slice(0, 14)}…` : label}
              </text>
            ))}
        </svg>
      </div>
      <div className="mt-4 flex items-center justify-end gap-3 text-[11px] font-medium text-slate-400">
        <span>{minimum.toFixed(3)}</span>
        <span
          className={`h-2.5 w-36 rounded-full ${
            diverging ? "diverging-legend" : "viridis-legend"
          }`}
        />
        <span>{maximum.toFixed(3)}</span>
      </div>
    </div>
  );
}

export function LossChart({
  data,
}: {
  data: { epoch: number; loss: number }[];
}) {
  const points = data.map((point) => ({
    epoch: point.epoch,
    loss: Math.max(point.loss, 1e-8),
  }));
  return (
    <ResponsiveContainer height={330} width="100%">
      <LineChart data={points} margin={{ top: 12, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="4 5" vertical={false} />
        <XAxis
          axisLine={false}
          dataKey="epoch"
          tick={AXIS}
          tickLine={false}
        />
        <YAxis
          axisLine={false}
          domain={["auto", "auto"]}
          scale="log"
          tick={AXIS}
          tickLine={false}
          width={62}
        />
        <Tooltip
          contentStyle={TOOLTIP_STYLE}
          cursor={{ stroke: "#14b8a6", strokeDasharray: "4 4" }}
        />
        <Line
          dataKey="loss"
          dot={false}
          stroke="#0f766e"
          strokeWidth={2.5}
          type="monotone"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function BarCompare({
  keys,
  before,
  after,
}: {
  keys: string[];
  before: number[];
  after: number[];
}) {
  const points = keys.map((key, index) => ({
    name: key,
    before: before[index],
    after: after[index],
  }));
  return (
    <ResponsiveContainer height={330} width="100%">
      <BarChart data={points} margin={{ top: 12, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="4 5" vertical={false} />
        <XAxis
          axisLine={false}
          dataKey="name"
          tick={AXIS}
          tickLine={false}
        />
        <YAxis axisLine={false} tick={AXIS} tickLine={false} width={56} />
        <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "#94a3b812" }} />
        <Legend
          iconType="circle"
          wrapperStyle={{ fontSize: "12px", paddingTop: "14px" }}
        />
        <Bar dataKey="before" fill="#f59e0b" radius={[6, 6, 0, 0]} />
        <Bar dataKey="after" fill="#0f766e" radius={[6, 6, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function Histogram({
  values,
  color = "#0f766e",
  n = 24,
}: {
  values: number[];
  color?: string;
  n?: number;
}) {
  if (!values.length) return null;
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const width = (maximum - minimum) / n || 1;
  const points = Array.from({ length: n }, (_, index) => ({
    range: (minimum + width * (index + 0.5)).toFixed(2),
    count: 0,
  }));
  values.forEach((value) => {
    let index = Math.floor((value - minimum) / width);
    index = Math.max(0, Math.min(n - 1, index));
    points[index].count += 1;
  });
  return (
    <ResponsiveContainer height={320} width="100%">
      <BarChart data={points} margin={{ top: 12, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="4 5" vertical={false} />
        <XAxis
          axisLine={false}
          dataKey="range"
          interval="preserveStartEnd"
          tick={AXIS}
          tickLine={false}
        />
        <YAxis
          allowDecimals={false}
          axisLine={false}
          tick={AXIS}
          tickLine={false}
          width={44}
        />
        <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "#94a3b812" }} />
        <Bar dataKey="count" fill={color} radius={[5, 5, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
