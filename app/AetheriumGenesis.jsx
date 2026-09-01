import React, { useState, useRef, useCallback } from "react";
import { motion } from "framer-motion";
import {
  Search,
  Camera,
  Map as MapIcon,
  Grid3x3,
  SlidersHorizontal,
  Check,
  X,
  Layers,
  Download,
  ChevronDown,
  ArrowLeftRight,
  Sparkles,
  ShieldCheck,
  Info,
  MousePointerClick,
  Hexagon,
  AlertTriangle,
  ClipboardCheck,
  HelpCircle,
  Upload,
  CheckCircle2,
  XCircle,
  ArrowLeft
} from "lucide-react";

/* ------------------------------------------------------------------ */
/* MOCK DATA                                                          */
/* ------------------------------------------------------------------ */

const QUERY_CHIPS = [
  "newly built structures near a river",
  "large vehicle concentrations on open ground",
  "cleared forest next to a road",
  "expanded port or dock infrastructure",
];

const TILE_RESULTS = [
  { id: "T-88421", date: "14 Aug 2025", match: 94, flagged: true },
  { id: "T-88437", date: "11 Aug 2025", match: 92, flagged: false },
  { id: "T-88502", date: "09 Aug 2025", match: 89, flagged: true },
  { id: "T-88519", date: "03 Aug 2025", match: 88, flagged: false },
  { id: "T-88560", date: "28 Jul 2025", match: 85, flagged: false },
  { id: "T-88574", date: "21 Jul 2025", match: 83, flagged: true },
];

const SIMILAR_SITES = [
  { id: "T-91204", place: "Northern river bend, 62km away", match: 88 },
  { id: "T-90188", place: "Eastern floodplain, 140km away", match: 84 },
  { id: "T-89765", place: "Coastal inlet, 205km away", match: 81 },
  { id: "T-88910", place: "Valley crossing, 38km away", match: 79 },
];

const INITIAL_REVIEW_QUEUE = [
  { id: "CD-0091", tile: "T-88421", type: "Construction / new infrastructure", confidence: 89, status: "pending", date: "14 Aug 2025" },
  { id: "CD-0092", tile: "T-88502", type: "Vehicle concentration on open ground", confidence: 76, status: "pending", date: "09 Aug 2025" },
  { id: "CD-0093", tile: "T-88574", type: "Vegetation clearance", confidence: 68, status: "confirmed", date: "21 Jul 2025" },
];

const PROVENANCE = {
  sceneId: "S2A_MSIL2A_20250814T083601_N0511_R064_T36RXV",
  coords: "31.5120° N, 34.4470° E",
  pipeline: "AGX-CHANGE v3.4.1 / ORTHO-CORR v2.1",
  sensor: "Sentinel-2A MSI, 10m resolution",
  baseline: "22 Mar 2024",
  current: "14 Aug 2025",
};

const TIMELINE_STEPS = [
  { date: "22 Mar 2024", label: "Baseline Scan", status: "Clean" },
  { date: "10 Nov 2024", label: "Initial Ground Prep", status: "Minor clearing" },
  { date: "05 Apr 2025", label: "Foundation Work", status: "Excavation" },
  { date: "14 Aug 2025", label: "Current Acquisition", status: "Structure Raised" },
];

const CLUSTERS = [
  { id: "Cluster-A", name: "River Basin Infrastructure", count: 14, x: 35, y: 40, active: true },
  { id: "Cluster-B", name: "Open Ground Depots", count: 8, x: 65, y: 30, active: false },
  { id: "Cluster-C", name: "Coastal Access Roads", count: 19, x: 50, y: 70, active: false },
];

/* ------------------------------------------------------------------ */
/* FRAMER MOTION ANIMATION VARIANTS                                   */
/* ------------------------------------------------------------------ */

const containerVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.06
    }
  }
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.3 } }
};

/* ------------------------------------------------------------------ */
/* 3D BACKGROUND WITH LARGER DETAILED SATELLITE                       */
/* ------------------------------------------------------------------ */

function Abstract3DGrid() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none z-0 bg-white" style={{ perspective: '1000px' }}>
      <style>{`
        @keyframes drift {
          0% { transform: rotateX(60deg) rotateZ(0deg); }
          100% { transform: rotateX(60deg) rotateZ(360deg); }
        }
        @keyframes satelliteMove {
          0% { transform: translate(-120px, -60px) rotate(-15deg); }
          100% { transform: translate(120px, 60px) rotate(-15deg); }
        }
        .grid-3d {
          position: absolute;
          top: -50%; left: -50%;
          width: 200%; height: 200%;
          background-image:
            linear-gradient(rgba(52, 211, 153, 0.15) 1px, transparent 1px),
            linear-gradient(90deg, rgba(52, 211, 153, 0.15) 1px, transparent 1px);
          background-size: 60px 60px;
          animation: drift 90s linear infinite;
          transform-origin: center center;
        }
        .anim-satellite {
          animation: satelliteMove 14s infinite alternate ease-in-out;
        }
        .fade-overlay {
          position: absolute;
          inset: 0;
          background: radial-gradient(circle at center, transparent 0%, #ffffff 75%);
        }
      `}</style>
      <div className="grid-3d" />
      
      {/* Larger, Highly Detailed Orbiting Satellite with Mint Green Boundary */}
      <div className="absolute top-1/4 right-1/4 anim-satellite z-10 opacity-85">
        <svg width="180" height="120" viewBox="0 0 180 120" fill="none">
          {/* Large Solar Panels with Grid Details */}
          <rect x="10" y="35" width="50" height="50" rx="4" fill="#E6FFFA" stroke="#10B981" strokeWidth="3" />
          <line x1="10" y1="51" x2="60" y2="51" stroke="#10B981" strokeWidth="1.5" />
          <line x1="10" y1="67" x2="60" y2="67" stroke="#10B981" strokeWidth="1.5" />
          <line x1="26" y1="35" x2="26" y2="85" stroke="#10B981" strokeWidth="1.5" />
          <line x1="44" y1="35" x2="44" y2="85" stroke="#10B981" strokeWidth="1.5" />

          <rect x="120" y="35" width="50" height="50" rx="4" fill="#E6FFFA" stroke="#10B981" strokeWidth="3" />
          <line x1="120" y1="51" x2="170" y2="51" stroke="#10B981" strokeWidth="1.5" />
          <line x1="120" y1="67" x2="170" y2="67" stroke="#10B981" strokeWidth="1.5" />
          <line x1="136" y1="35" x2="136" y2="85" stroke="#10B981" strokeWidth="1.5" />
          <line x1="154" y1="35" x2="154" y2="85" stroke="#10B981" strokeWidth="1.5" />

          {/* Central Body / Bus */}
          <rect x="60" y="25" width="60" height="70" rx="8" fill="#334155" stroke="#10B981" strokeWidth="3.5" />
          <circle cx="90" cy="60" r="14" fill="#10B981" fillOpacity="0.3" stroke="#10B981" strokeWidth="2.5" />
          <circle cx="90" cy="60" r="6" fill="#10B981" />

          {/* Connectors */}
          <rect x="58" y="55" width="4" height="10" fill="#10B981" />
          <rect x="118" y="55" width="4" height="10" fill="#10B981" />

          {/* Communication Dish */}
          <path d="M75,25 Q90,5 105,25 Z" fill="#E6FFFA" stroke="#10B981" strokeWidth="2.5" />
          <line x1="90" y1="25" x2="90" y2="15" stroke="#10B981" strokeWidth="2.5" />

          {/* Scanner / Sensor Beamcone */}
          <polygon points="75,95 45,120 135,120" fill="rgba(16, 185, 129, 0.2)" />
        </svg>
      </div>

      <div className="fade-overlay" />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* SHARED PRIMITIVES                                                  */
/* ------------------------------------------------------------------ */

const Panel = ({ children, className = "" }) => (
  <div
    className={`relative rounded-xl border backdrop-blur-xl ${className}`}
    style={{ background: "rgba(255, 255, 255, 0.9)", borderColor: "rgba(16, 185, 129, 0.25)", boxShadow: "0 4px 20px rgba(0,0,0,0.03)" }}
  >
    {children}
  </div>
);

const PanelHeader = ({ icon: Icon, title, hint, right }) => (
  <div className="px-5 py-3.5 border-b flex items-start justify-between gap-3" style={{ borderColor: "rgba(16, 185, 129, 0.15)" }}>
    <div className="flex items-start gap-2.5">
      {Icon && (
        <div className="mt-0.5 flex items-center justify-center w-6 h-6 rounded-md flex-shrink-0" style={{ background: "rgba(16, 185, 129, 0.15)" }}>
          <Icon size={13} className="text-emerald-600" strokeWidth={2} />
        </div>
      )}
      <div>
        <div className="text-[13.5px] font-semibold text-slate-800">{title}</div>
        {hint && <div className="text-[12px] text-slate-500 mt-0.5">{hint}</div>}
      </div>
    </div>
    {right}
  </div>
);

function Tip({ children }) {
  return (
    <div
      className="flex items-start gap-2 px-3.5 py-2.5 rounded-lg text-[12.5px] text-slate-600"
      style={{ background: "rgba(16, 185, 129, 0.08)", border: "1px solid rgba(16, 185, 129, 0.25)" }}
    >
      <Info size={14} className="text-emerald-600 flex-shrink-0 mt-0.5" />
      <span>{children}</span>
    </div>
  );
}

const Logo = ({ size = 22, onClick }) => (
  <button onClick={onClick} className="flex items-center gap-2.5 group">
    <div
      className="flex items-center justify-center rounded-lg transition-transform group-hover:scale-105"
      style={{
        width: size + 14,
        height: size + 14,
        background: "linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(52, 211, 153, 0.3))",
        border: "1px solid rgba(16, 185, 129, 0.4)",
      }}
    >
      <Hexagon size={size} className="text-emerald-700" strokeWidth={2} />
    </div>
    <span className="text-[16px] font-semibold text-slate-800 tracking-tight">Aetherium Genesis</span>
  </button>
);

/* ------------------------------------------------------------------ */
/* HOME                                                               */
/* ------------------------------------------------------------------ */

function HomeSearch({ query, setQuery, onSearch }) {
  const [showImage, setShowImage] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [fileName, setFileName] = useState(null);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) setFileName(f.name);
  };

  return (
    <div className="relative min-h-screen w-full flex items-center justify-center px-6">
      <Abstract3DGrid />

      <div className="relative z-10 w-full max-w-2xl -mt-16 flex flex-col items-center">
        <div className="flex items-center justify-center w-16 h-16 rounded-2xl mb-5 shadow-sm" style={{ background: "linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(52, 211, 153, 0.2))", border: "1px solid rgba(16, 185, 129, 0.4)" }}>
          <Hexagon size={30} className="text-emerald-600" strokeWidth={1.5} />
        </div>
        <h1 className="text-[32px] sm:text-[38px] font-semibold text-slate-800 tracking-tight text-center">Aetherium Genesis</h1>
        <p className="text-[14px] text-slate-500 mt-2 text-center max-w-md">
          Search the satellite archive and see what's changed on the ground.
        </p>

        <div
          className="w-full mt-8 flex items-center gap-3 px-5 py-4 rounded-full border shadow-xl"
          style={{ borderColor: "rgba(16, 185, 129, 0.4)", background: "rgba(255, 255, 255, 0.85)", backdropFilter: "blur(12px)" }}
        >
          <Search size={17} className="text-emerald-600 flex-shrink-0" />
          <input
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && onSearch()}
            placeholder="Search the imagery archive…"
            className="flex-1 bg-transparent outline-none text-[15px] text-slate-800 placeholder:text-slate-400"
          />
          <button
            onClick={() => setShowImage((s) => !s)}
            title="Search using an image instead"
            className="flex items-center justify-center w-8 h-8 rounded-full transition-colors flex-shrink-0"
            style={{ background: showImage ? "rgba(16, 185, 129, 0.2)" : "transparent", color: showImage ? "#047857" : "#64748b" }}
          >
            <Camera size={16} />
          </button>
        </div>

        {showImage && (
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            className="w-full mt-3 flex items-center gap-3 px-4 py-3 rounded-xl border border-dashed cursor-pointer transition-colors"
            style={{
              borderColor: dragOver ? "rgba(16, 185, 129, 0.8)" : "rgba(16, 185, 129, 0.4)",
              background: dragOver ? "rgba(16, 185, 129, 0.1)" : "rgba(255, 255, 255, 0.6)",
            }}
          >
            <Upload size={15} className="text-emerald-600 flex-shrink-0" />
            <div className="text-[12px] leading-tight">
              <div className="text-slate-700 font-medium">{fileName || "Drop a photo or scene here"}</div>
              <div className="text-slate-500">We'll find visually similar locations in the archive</div>
            </div>
          </div>
        )}

        <div className="flex items-center gap-3 mt-6">
          <button
            onClick={onSearch}
            className="px-5 py-2.5 rounded-lg text-[13px] font-semibold transition-colors shadow-sm"
            style={{ background: "#10B981", color: "#FFFFFF", border: "1px solid #059669" }}
          >
            Search Archive
          </button>
        </div>

        <div className="flex flex-wrap items-center justify-center gap-x-1.5 gap-y-1 mt-6 text-[12px] max-w-lg">
          <span className="text-slate-500 mr-0.5">Try:</span>
          {QUERY_CHIPS.map((c, i) => (
            <React.Fragment key={c}>
              <button onClick={() => setQuery(c)} className="text-emerald-600 hover:text-emerald-800 transition-colors underline decoration-emerald-200 underline-offset-2">
                {c}
              </button>
              {i < QUERY_CHIPS.length - 1 && <span className="text-slate-300">·</span>}
            </React.Fragment>
          ))}
        </div>
      </div>

      <div className="absolute bottom-6 left-0 right-0 flex items-center justify-center gap-2 text-[11.5px] text-slate-500 z-10">
        <span className="relative flex h-1.5 w-1.5">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-60" style={{ background: "#10B981" }} />
          <span className="relative inline-flex rounded-full h-1.5 w-1.5" style={{ background: "#10B981" }} />
        </span>
        Running on local hardware · 1,240 tiles indexed · nothing leaves this network
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* COMPACT TOP SEARCH BAR                                             */
/* ------------------------------------------------------------------ */

function TopSearchBar({ query, setQuery, onSearch, onHome, filtersOpen, setFiltersOpen }) {
  const [showHealth, setShowHealth] = useState(false);

  return (
    <div className="sticky top-0 z-20 border-b backdrop-blur-xl" style={{ borderColor: "rgba(16, 185, 129, 0.2)", background: "rgba(255, 255, 255, 0.9)" }}>
      <div className="max-w-[1500px] mx-auto px-6 py-3 flex items-center gap-4 justify-between">
        <div className="flex items-center gap-3 flex-1">
          <button
            onClick={onHome}
            title="Back to home"
            className="flex items-center justify-center w-8 h-8 rounded-lg border bg-white hover:bg-slate-50 transition-colors text-slate-600 flex-shrink-0"
            style={{ borderColor: "rgba(16, 185, 129, 0.3)" }}
          >
            <ArrowLeft size={15} className="text-emerald-600" />
          </button>
          <Logo size={17} onClick={onHome} />
          <div
            className="flex-1 flex items-center gap-2.5 px-4 py-2 rounded-full border max-w-xl"
            style={{ borderColor: "rgba(16, 185, 129, 0.3)", background: "rgba(255, 255, 255, 1)" }}
          >
            <Search size={14} className="text-emerald-500 flex-shrink-0" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && onSearch()}
              className="flex-1 bg-transparent outline-none text-[13px] text-slate-800"
            />
            <button onClick={onSearch} className="text-[11.5px] font-medium text-emerald-600 flex-shrink-0">Search</button>
          </div>
          <button
            onClick={() => setFiltersOpen((o) => !o)}
            className="hidden sm:flex items-center gap-1.5 px-3 py-2 rounded-lg border text-[12px] text-slate-600 transition-colors flex-shrink-0 bg-white"
            style={{ borderColor: "rgba(16, 185, 129, 0.25)" }}
          >
            <SlidersHorizontal size={12} className="text-emerald-500" />
            Filters
            <ChevronDown size={12} className={`text-slate-400 transition-transform ${filtersOpen ? "rotate-180" : ""}`} />
          </button>
        </div>

        <div className="relative">
          <button 
            onClick={() => setShowHealth(!showHealth)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-full border bg-emerald-50 border-emerald-200 text-xs font-medium text-emerald-700 transition-all hover:bg-emerald-100"
          >
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            Local Network: Air-Gapped
          </button>

          {showHealth && (
            <div className="absolute right-0 mt-2 w-64 p-4 rounded-xl bg-white border border-slate-200 shadow-xl text-xs space-y-2 z-30">
              <div className="font-semibold text-slate-800 border-b pb-1">Offline System Status</div>
              <div className="flex justify-between text-slate-600"><span>Vector Index Size:</span> <span className="font-mono text-emerald-600">4.2 GB</span></div>
              <div className="flex justify-between text-slate-600"><span>Unprocessed Tiles:</span> <span className="font-mono text-slate-800">12</span></div>
              <div className="flex justify-between text-slate-600"><span>GPU VRAM Usage:</span> <span className="font-mono text-slate-800">60%</span></div>
              <div className="flex justify-between text-slate-600"><span>External API Calls:</span> <span className="font-mono text-emerald-600">0 (Blocked)</span></div>
            </div>
          )}
        </div>
      </div>

      {filtersOpen && (
        <div className="max-w-[1500px] mx-auto px-6 pb-3.5 grid grid-cols-2 lg:grid-cols-4 gap-2.5">
          {[
            { label: "Date range", value: "Jan 2024 – Aug 2025" },
            { label: "Max cloud cover", value: "15% or less" },
            { label: "Sensor", value: "Sentinel-2" },
            { label: "Area of interest", value: "Drawn on map" },
          ].map((f) => (
            <div key={f.label} className="px-3 py-2.5 rounded-md border bg-white" style={{ borderColor: "rgba(16, 185, 129, 0.2)" }}>
              <div className="text-[10.5px] text-slate-500">{f.label}</div>
              <div className="text-[12.5px] text-slate-800 font-medium mt-0.5">{f.value}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* PROCEDURAL SCENE SWATCH                                            */
/* ------------------------------------------------------------------ */

function SceneSwatch({ seed = 0, variant = "before", showMask = false, className = "" }) {
  const hue = 140 + ((seed * 37) % 30);
  return (
    <div
      className={`relative overflow-hidden ${className}`}
      style={{
        background:
          variant === "before"
            ? `radial-gradient(circle at 30% 30%, hsl(${hue},30%,85%), #F0FDF4 70%)`
            : `radial-gradient(circle at 65% 40%, hsl(${hue + 10},35%,80%), #E6FFFA 70%)`,
      }}
    >
      {[...Array(5)].map((_, i) => (
        <div
          key={i}
          className="absolute rounded-sm"
          style={{
            left: `${(seed * 13 + i * 21) % 80}%`,
            top: `${(seed * 27 + i * 17) % 75}%`,
            width: `${8 + ((seed + i) % 4) * 6}px`,
            height: `${8 + ((seed + i * 2) % 4) * 6}px`,
            background: variant === "after" && i % 2 === 0 ? "rgba(16, 185, 129, 0.4)" : "rgba(148, 163, 184, 0.4)",
            transform: `rotate(${(i * 23) % 45}deg)`,
          }}
        />
      ))}
      {variant === "after" && showMask && (
        <div
          className="absolute rounded-sm"
          style={{
            left: "42%",
            top: "38%",
            width: "26%",
            height: "22%",
            background: "rgba(239, 68, 68, 0.15)",
            border: "1.5px solid rgba(239, 68, 68, 0.7)",
            boxShadow: "0 0 10px rgba(239, 68, 68, 0.2)",
          }}
        />
      )}
      <div className="absolute bottom-1.5 left-2 text-[10px] text-slate-700 font-medium bg-white/70 px-1.5 py-0.5 rounded shadow-sm">
        {variant === "before" ? "Before" : "After"}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* RESULTS                                                            */
/* ------------------------------------------------------------------ */

function TileCard({ tile, idx, active, onSelect }) {
  return (
    <motion.div variants={itemVariants}>
      <motion.button
        whileHover={{ scale: 1.02, y: -2 }}
        whileTap={{ scale: 0.98 }}
        transition={{ type: "spring", stiffness: 400, damping: 17 }}
        onClick={() => onSelect(tile)}
        className="group relative rounded-lg border overflow-hidden text-left transition-all bg-white w-full"
        style={{
          borderColor: active ? "rgba(16, 185, 129, 0.8)" : "rgba(16, 185, 129, 0.2)",
          boxShadow: active ? "0 0 0 1px rgba(16, 185, 129, 0.5), 0 4px 12px rgba(16, 185, 129, 0.1)" : "none",
        }}
      >
        <SceneSwatch seed={idx} variant="before" className="h-28 w-full" />
        {tile.flagged && (
          <div
            className="absolute top-2 right-2 flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium"
            style={{ background: "#EF4444", color: "#fff", boxShadow: "0 2px 4px rgba(239,68,68,0.2)" }}
          >
            <AlertTriangle size={9} /> Change found
          </div>
        )}
        <div className="px-3 py-2.5 border-t" style={{ borderColor: "rgba(16, 185, 129, 0.15)", background: "#FAFAFA" }}>
          <div className="flex items-center justify-between">
            <span className="text-[11.5px] font-medium text-slate-700">{tile.id}</span>
            <span className="text-[11.5px] font-semibold text-emerald-600">{tile.match}% match</span>
          </div>
          <div className="text-[10.5px] text-slate-500 mt-0.5">{tile.date}</div>
        </div>
      </motion.button>
    </motion.div>
  );
}

function ResultsSection({ view, setView, selected, onSelect }) {
  return (
    <Panel>
      <PanelHeader
        icon={Grid3x3}
        title="Results & Semantic Discovery"
        hint="Switch between Grid, Geographic Map, or Unsupervised Clusters"
        right={
          <div className="flex items-center gap-1 rounded-md border p-0.5 bg-white" style={{ borderColor: "rgba(16, 185, 129, 0.2)" }}>
            <button
              onClick={() => setView("grid")}
              className="px-2.5 py-1.5 rounded text-[11.5px] font-medium transition-colors"
              style={{ background: view === "grid" ? "rgba(16, 185, 129, 0.15)" : "transparent", color: view === "grid" ? "#047857" : "#64748b" }}
            >
              Grid
            </button>
            <button
              onClick={() => setView("map")}
              className="px-2.5 py-1.5 rounded text-[11.5px] font-medium transition-colors"
              style={{ background: view === "map" ? "rgba(16, 185, 129, 0.15)" : "transparent", color: view === "map" ? "#047857" : "#64748b" }}
            >
              Map
            </button>
            <button
              onClick={() => setView("clusters")}
              className="px-2.5 py-1.5 rounded text-[11.5px] font-medium transition-colors"
              style={{ background: view === "clusters" ? "rgba(16, 185, 129, 0.15)" : "transparent", color: view === "clusters" ? "#047857" : "#64748b" }}
            >
              Clusters
            </button>
          </div>
        }
      />
      <div className="p-4">
        {view === "grid" && (
          <motion.div 
            variants={containerVariants}
            initial="hidden"
            animate="show"
            className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-3 gap-3"
          >
            {TILE_RESULTS.map((t, i) => (
              <TileCard key={t.id} tile={t} idx={i} active={selected?.id === t.id} onSelect={onSelect} />
            ))}
          </motion.div>
        )}

        {view === "map" && (
          <div className="relative h-80 w-full overflow-hidden rounded-lg border border-emerald-100 bg-emerald-50/20">
            {TILE_RESULTS.map((t, i) => (
              <button
                key={t.id}
                onClick={() => onSelect(t)}
                className="absolute flex flex-col items-center group"
                style={{ left: `${18 + ((i * 13) % 62)}%`, top: `${20 + ((i * 19) % 55)}%` }}
              >
                <div
                  className="relative flex items-center justify-center w-5 h-5 rounded-full transition-transform group-hover:scale-125"
                  style={{
                    background: t.flagged ? "#EF4444" : "#10B981",
                    boxShadow: selected?.id === t.id ? "0 0 0 3px rgba(16, 185, 129, 0.4)" : "0 2px 4px rgba(0,0,0,0.15)",
                  }}
                />
                <span className="mt-1 text-[10px] font-semibold text-slate-700 bg-white px-1.5 py-0.5 rounded shadow-sm">{t.id}</span>
              </button>
            ))}
          </div>
        )}

        {view === "clusters" && (
          <div className="relative h-80 w-full overflow-hidden rounded-lg border border-emerald-200 bg-gradient-to-br from-emerald-50/50 to-white p-4 flex flex-col items-center justify-center">
            <div className="absolute top-3 left-3 text-xs font-semibold text-emerald-800 flex items-center gap-1.5">
              <Sparkles size={13} className="text-emerald-600" /> Unsupervised Embedding Vector Grouping (HDBSCAN)
            </div>
            <div className="relative w-full h-full">
              {CLUSTERS.map((c) => (
                <div
                  key={c.id}
                  className="absolute p-3 rounded-xl border bg-white shadow-md transition-all hover:border-emerald-500 cursor-pointer flex flex-col items-center"
                  style={{ left: `${c.x}%`, top: `${c.y}%`, borderColor: c.active ? "#10B981" : "#E2E8F0" }}
                >
                  <div className="w-8 h-8 rounded-full bg-emerald-100 flex items-center justify-center font-bold text-emerald-700 text-xs mb-1">
                    {c.count}
                  </div>
                  <span className="text-[11px] font-medium text-slate-800 whitespace-nowrap">{c.name}</span>
                  <span className="text-[9px] text-slate-400">Auto-discovered cluster</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </Panel>
  );
}

/* ------------------------------------------------------------------ */
/* CHANGE COMPARISON                                                  */
/* ------------------------------------------------------------------ */

function ChangeSlider({ seed }) {
  const [split, setSplit] = useState(50);
  const [showMask, setShowMask] = useState(true);
  const [activeStep, setActiveStep] = useState(3);
  const ref = useRef(null);

  const handleMove = useCallback((clientX) => {
    if (!ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    const pct = Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100));
    setSplit(pct);
  }, []);

  return (
    <div className="space-y-4">
      <div
        ref={ref}
        className="relative h-52 w-full rounded-lg overflow-hidden cursor-ew-resize select-none border"
        style={{ borderColor: "rgba(16, 185, 129, 0.3)" }}
        onMouseDown={(e) => {
          handleMove(e.clientX);
          const onMove = (ev) => handleMove(ev.clientX);
          const onUp = () => {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onUp);
          };
          window.addEventListener("mousemove", onMove);
          window.addEventListener("mouseup", onUp);
        }}
      >
        <SceneSwatch seed={seed} variant="after" showMask={showMask} className="absolute inset-0" />
        <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}>
          <SceneSwatch seed={seed} variant="before" className="absolute inset-0" />
        </div>
        <div className="absolute top-0 bottom-0 w-0.5" style={{ left: `${split}%`, background: "#10B981", boxShadow: "0 0 8px rgba(16, 185, 129, 0.6)" }}>
          <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 flex items-center justify-center w-7 h-7 rounded-full shadow-md" style={{ background: "#10B981" }}>
            <ArrowLeftRight size={13} className="text-white" />
          </div>
        </div>
      </div>

      <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg space-y-2">
        <div className="flex items-center justify-between text-xs font-semibold text-slate-700">
          <span>Multi-Temporal Progression</span>
          <span className="text-emerald-600">{TIMELINE_STEPS[activeStep].date} ({TIMELINE_STEPS[activeStep].label})</span>
        </div>
        <div className="relative flex items-center justify-between px-2">
          <div className="absolute left-4 right-4 h-0.5 bg-slate-200 z-0" />
          {TIMELINE_STEPS.map((step, idx) => (
            <button
              key={step.date}
              onClick={() => setActiveStep(idx)}
              className="relative z-10 flex flex-col items-center group transition-all"
            >
              <div 
                className={`w-4 h-4 rounded-full border-2 transition-all ${
                  activeStep === idx ? 'bg-emerald-500 border-white ring-2 ring-emerald-400' : 'bg-white border-slate-300 hover:border-emerald-400'
                }`} 
              />
              <span className="text-[10px] text-slate-500 mt-1 whitespace-nowrap">{step.date}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1.5 text-[11.5px] text-slate-500">
          <MousePointerClick size={12} className="text-emerald-500" /> Drag divider to compare
        </span>
        <button
          onClick={() => setShowMask((s) => !s)}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-[11.5px] font-medium border transition-colors bg-white"
          style={{
            borderColor: showMask ? "rgba(239, 68, 68, 0.3)" : "rgba(226, 232, 240, 1)",
            color: showMask ? "#DC2626" : "#64748b",
            background: showMask ? "rgba(239, 68, 68, 0.05)" : "transparent",
          }}
        >
          <Layers size={12} /> {showMask ? "Hide mask" : "Show mask"}
        </button>
      </div>
    </div>
  );
}

function MetricCard({ label, value, tone }) {
  const colorMap = { emerald: "#059669", cyan: "#047857", amber: "#D97706", slate: "#475569" };
  return (
    <div className="px-3 py-2.5 rounded-md border bg-white" style={{ borderColor: "rgba(16, 185, 129, 0.15)" }}>
      <div className="text-[10.5px] text-slate-500">{label}</div>
      <div className="text-[13px] font-semibold mt-0.5" style={{ color: colorMap[tone || "slate"] }}>{value}</div>
    </div>
  );
}

function ChangeAnalysisSection({ selected, onFindSimilar, showingSimilar }) {
  if (!selected) {
    return (
      <Panel className="p-8 flex flex-col items-center justify-center text-center gap-2 h-full bg-white/50">
        <div className="flex items-center justify-center w-11 h-11 rounded-full mb-1" style={{ background: "rgba(16, 185, 129, 0.1)" }}>
          <MousePointerClick size={18} className="text-emerald-600" />
        </div>
        <div className="text-[13.5px] font-semibold text-slate-700">No result selected yet</div>
        <div className="text-[12.5px] text-slate-500 max-w-xs">Pick a tile from the results to see multi-temporal change history.</div>
      </Panel>
    );
  }

  return (
    <Panel>
      <PanelHeader icon={ArrowLeftRight} title="How has this location changed?" hint={`Showing ${selected.id}`} />
      <div className="p-4">
        <ChangeSlider seed={TILE_RESULTS.findIndex((t) => t.id === selected.id)} />

        <div className="grid grid-cols-2 gap-2.5 mt-4">
          <MetricCard label="What changed" value="New construction" tone="cyan" />
          <MetricCard label="Confidence" value="89% — high" tone="emerald" />
          <MetricCard label="Cloud & shadow noise" value="Filtered out" tone="amber" />
          <MetricCard label="First observed" value={selected.date} />
        </div>

        <button
          onClick={onFindSimilar}
          className="w-full flex items-center justify-center gap-2 mt-4 px-3 py-2.5 rounded-md text-[12.5px] font-semibold transition-colors shadow-sm"
          style={{
            background: showingSimilar ? "rgba(16, 185, 129, 0.15)" : "#F8FAFC",
            border: "1px solid rgba(16, 185, 129, 0.4)",
            color: "#047857",
          }}
        >
          <Sparkles size={14} /> {showingSimilar ? "Hide similar sites" : "Find similar sites in the archive"}
        </button>
      </div>
    </Panel>
  );
}

/* ------------------------------------------------------------------ */
/* SIMILAR SITES                                                      */
/* ------------------------------------------------------------------ */

function SimilarSites({ selected, onSelectSite }) {
  if (!selected) return null;
  return (
    <Panel>
      <PanelHeader
        icon={Sparkles}
        title="Similar sites"
        hint={`Other locations across the archive that look like ${selected.id}`}
      />
      <div className="p-4 grid grid-cols-2 sm:grid-cols-4 gap-3">
        {SIMILAR_SITES.map((s, i) => (
          <button
            key={s.id}
            onClick={() => onSelectSite({ id: s.id, date: selected.date, match: s.match, flagged: false })}
            className="group relative rounded-lg border overflow-hidden text-left transition-colors bg-white hover:border-emerald-400"
            style={{ borderColor: "rgba(16, 185, 129, 0.25)" }}
          >
            <SceneSwatch seed={i + 20} variant="before" className="h-24 w-full" />
            <div className="px-3 py-2.5 border-t" style={{ borderColor: "rgba(16, 185, 129, 0.15)", background: "#FAFAFA" }}>
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-medium text-slate-700">{s.id}</span>
                <span className="text-[11px] font-semibold text-emerald-600">{s.match}%</span>
              </div>
              <div className="text-[10.5px] text-slate-500 mt-0.5">{s.place}</div>
            </div>
          </button>
        ))}
      </div>
    </Panel>
  );
}

/* ------------------------------------------------------------------ */
/* REVIEW QUEUE                                                       */
/* ------------------------------------------------------------------ */

function ReviewQueue() {
  const [items, setItems] = useState(INITIAL_REVIEW_QUEUE);
  const setStatus = (id, status) => setItems((prev) => prev.map((it) => (it.id === id ? { ...it, status } : it)));

  return (
    <Panel>
      <PanelHeader
        icon={ClipboardCheck}
        title="Analyst Review Queue & Audit Trail"
        hint="Confirm real changes or reject false alarms. Decisions are saved to local audit trail."
        right={<span className="text-[11.5px] font-medium text-slate-500">{items.filter((i) => i.status === "pending").length} pending verification</span>}
      />
      <div className="p-4 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-slate-50/70 border border-slate-200 rounded-lg p-3 space-y-3">
          <div className="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center justify-between">
            <span>Pending Review</span>
            <span className="bg-amber-100 text-amber-800 px-2 py-0.5 rounded-full text-[10px]">
              {items.filter(i => i.status === 'pending').length}
            </span>

          </div>
          {items.filter(i => i.status === 'pending').map((it) => (
            <div key={it.id} className="bg-white p-3 rounded-md border border-slate-200 shadow-sm space-y-2">
              <div className="flex justify-between items-center text-xs">
                <span className="font-mono font-semibold text-emerald-600">{it.id}</span>
                <span className="text-slate-400">{it.date}</span>
              </div>
              <p className="text-xs font-medium text-slate-800">{it.type}</p>
              <div className="flex items-center justify-between pt-2 border-t border-slate-100">
                <span className="text-[10px] text-slate-500">Conf: {it.confidence}%</span>
                <div className="flex gap-1">
                  <button 
                    onClick={() => setStatus(it.id, 'confirmed')} 
                    className="p-1 rounded bg-emerald-50 text-emerald-600 hover:bg-emerald-100 transition-colors"
                    title="Confirm Change"
                  >
                    <CheckCircle2 size={15} />
                  </button>
                  <button 
                    onClick={() => setStatus(it.id, 'rejected')} 
                    className="p-1 rounded bg-red-50 text-red-600 hover:bg-red-100 transition-colors"
                    title="Reject False Alarm"
                  >
                    <XCircle size={15} />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="bg-emerald-50/30 border border-emerald-100 rounded-lg p-3 space-y-3">
          <div className="text-xs font-bold text-emerald-800 uppercase tracking-wider flex items-center justify-between">
            <span>Confirmed Audit Log</span>
            <span className="bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded-full text-[10px]">
              {items.filter(i => i.status === 'confirmed').length}
            </span>

          </div>
          {items.filter(i => i.status === 'confirmed').map((it) => (
            <div key={it.id} className="bg-white p-3 rounded-md border border-emerald-200 shadow-sm space-y-1">
              <div className="flex justify-between items-center text-xs">
                <span className="font-mono font-semibold text-emerald-700">{it.id}</span>
                <span className="text-emerald-600 text-[10px] bg-emerald-50 px-1.5 py-0.5 rounded">Verified</span>
              </div>
              <p className="text-xs font-medium text-slate-800">{it.type}</p>
              <span className="text-[10px] text-slate-400">Processed from tile {it.tile}</span>
            </div>
          ))}
        </div>

        <div className="bg-red-50/30 border border-red-100 rounded-lg p-3 space-y-3">
          <div className="text-xs font-bold text-red-800 uppercase tracking-wider flex items-center justify-between">
            <span>Rejected / Suppressed</span>
            <span className="bg-red-100 text-red-800 px-2 py-0.5 rounded-full text-[10px]">
              {items.filter(i => i.status === 'rejected').length}
            </span>

          </div>
          {items.filter(i => i.status === 'rejected').map((it) => (
            <div key={it.id} className="bg-white p-3 rounded-md border border-red-200 shadow-sm space-y-1">
              <div className="flex justify-between items-center text-xs">
                <span className="font-mono font-semibold text-red-700">{it.id}</span>
                <span className="text-red-600 text-[10px] bg-red-50 px-1.5 py-0.5 rounded">False Alarm</span>
              </div>
              <p className="text-xs font-medium text-slate-800">{it.type}</p>
              <span className="text-[10px] text-slate-400">Suppressed by quality filter</span>
            </div>
          ))}
        </div>
      </div>
    </Panel>
  );
}

/* ------------------------------------------------------------------ */
/* PROVENANCE                                                         */
/* ------------------------------------------------------------------ */

function ProvenanceDisclosure() {
  const [open, setOpen] = useState(false);
  const [cloudMask, setCloudMask] = useState(true);
  const [norm, setNorm] = useState(true);

  return (
    <>
      <Panel>
        <button onClick={() => setOpen(true)} className="w-full flex items-center justify-between px-5 py-3.5 bg-white/40 hover:bg-slate-50/50 transition-colors">
          <div className="flex items-center gap-2.5">
            <HelpCircle size={14} className="text-slate-400" />
            <span className="text-[13px] font-medium text-slate-700">Open Provenance & Quality Settings Drawer</span>
          </div>
          <ChevronDown size={14} className="text-slate-400 -rotate-90" />
        </button>
      </Panel>

      {open && (
        <div className="fixed inset-0 z-50 overflow-hidden bg-slate-900/20 backdrop-blur-sm flex justify-end">
          <div className="w-full max-w-md bg-white h-full shadow-2xl p-6 flex flex-col justify-between overflow-y-auto border-l border-slate-200">
            <div className="space-y-6">
              <div className="flex items-center justify-between border-b pb-4">
                <h3 className="text-base font-bold text-slate-800">Geospatial Provenance & Quality</h3>
                <button onClick={() => setOpen(false)} className="p-1 rounded-full hover:bg-slate-100 text-slate-500">
                  <X size={18} />
                </button>
              </div>

              <div className="space-y-3 bg-slate-50 p-4 rounded-xl border border-slate-200">
                <h4 className="text-xs font-semibold text-slate-700 uppercase">False-Alarm Suppression Controls</h4>
                <label className="flex items-center justify-between text-xs text-slate-700 cursor-pointer">
                  <span>Cloud & Shadow Masking (SCL)</span>
                  <input type="checkbox" checked={cloudMask} onChange={() => setCloudMask(!cloudMask)} className="accent-emerald-500 w-4 h-4" />
                </label>
                <label className="flex items-center justify-between text-xs text-slate-700 cursor-pointer">
                  <span>Radiometric Normalization</span>
                  <input type="checkbox" checked={norm} onChange={() => setNorm(!norm)} className="accent-emerald-500 w-4 h-4" />
                </label>
              </div>

              <div className="space-y-2 text-xs">
                <h4 className="font-semibold text-slate-700 uppercase">STAC Metadata Record</h4>
                <div className="bg-slate-900 text-emerald-400 p-3 rounded-lg font-mono text-[11px] overflow-x-auto space-y-1">
                  <div>{`{`}</div>
                  <div className="pl-3">{`"scene_id": "${PROVENANCE.sceneId}",`}</div>
                  <div className="pl-3">{`"coordinates": "${PROVENANCE.coords}",`}</div>
                  <div className="pl-3">{`"pipeline": "${PROVENANCE.pipeline}",`}</div>
                  <div className="pl-3">{`"cloud_mask_applied": ${cloudMask},`}</div>
                  <div className="pl-3">{`"normalization": ${norm}`}</div>
                  <div>{`}`}</div>
                </div>
              </div>
            </div>

            <button
              onClick={() => setOpen(false)}
              className="w-full flex items-center justify-center gap-2 mt-6 px-4 py-3 rounded-xl text-xs font-bold transition-colors bg-emerald-600 text-white shadow-sm hover:bg-emerald-700"
            >
              <Download size={14} /> Export GeoTIFF & Audit Report
            </button>
          </div>
        </div>
      )}
    </>
  );
}

/* ------------------------------------------------------------------ */
/* ROOT COMPONENT                                                     */
/* ------------------------------------------------------------------ */

export default function AetheriumGenesis() {
  const [query, setQuery] = useState("");
  const [phase, setPhase] = useState("home"); 
  const [view, setView] = useState("grid");
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [selected, setSelected] = useState(null);
  const [showSimilar, setShowSimilar] = useState(false);

  const onSearch = () => setPhase("results");
  const onHome = () => {
    setPhase("home");
    setSelected(null);
    setShowSimilar(false);
  };
  const onSelect = (tile) => {
    setSelected(tile);
    setShowSimilar(false);
  };

  const baseStyle = {
    background: "#F8FAFC",
    fontFamily: "'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif",
  };

  if (phase === "home") {
    return (
      <div className="min-h-screen w-full text-slate-800" style={{ background: "#FFFFFF", fontFamily: baseStyle.fontFamily }}>
        <style>{`::-webkit-scrollbar { width: 8px; } ::-webkit-scrollbar-thumb { background: rgba(16, 185, 129, 0.25); border-radius: 4px; }`}</style>
        <HomeSearch query={query} setQuery={setQuery} onSearch={onSearch} />
      </div>
    );
  }

  return (
    <div className="min-h-screen w-full text-slate-800" style={baseStyle}>
      <style>{`::-webkit-scrollbar { width: 8px; } ::-webkit-scrollbar-thumb { background: rgba(16, 185, 129, 0.25); border-radius: 4px; }`}</style>

      <TopSearchBar query={query} setQuery={setQuery} onSearch={onSearch} onHome={onHome} filtersOpen={filtersOpen} setFiltersOpen={setFiltersOpen} />

      <motion.main 
        initial={{ opacity: 0, y: 15 }} 
        animate={{ opacity: 1, y: 0 }} 
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="max-w-[1500px] mx-auto px-6 py-4 space-y-4"
      >
        {!selected && <Tip>Select a tile from the results to examine temporal changes and time-series progression.</Tip>}

        <div className="grid grid-cols-1 xl:grid-cols-5 gap-4 items-start">
          <div className="xl:col-span-3">
            <ResultsSection view={view} setView={setView} selected={selected} onSelect={onSelect} />
          </div>
          <div className="xl:col-span-2">
            <ChangeAnalysisSection selected={selected} onFindSimilar={() => setShowSimilar((s) => !s)} showingSimilar={showSimilar} />
          </div>
        </div>

        {showSimilar && <SimilarSites selected={selected} onSelectSite={onSelect} />}

        <ReviewQueue />
        <ProvenanceDisclosure />

        <div className="flex items-center gap-2 text-[11.5px] font-medium text-slate-500 pt-1 pb-4">
          <ShieldCheck size={13} className="text-emerald-500" />
          Everything runs on local hardware — nothing leaves this network.
        </div>
      </motion.main>
    </div>
  );
}
```[cite: 2]
