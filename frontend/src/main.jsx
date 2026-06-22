import React from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  ChevronRight,
  GitBranch,
  ImageUp,
  Layers,
  Map,
  Network,
  RefreshCw,
  Route,
  ShieldCheck,
  Upload,
  Wrench,
} from "lucide-react";
import "./styles.css";

const API_BASE = "http://127.0.0.1:8000";

const pages = [
  { id: "overview", label: "Overview", icon: Activity },
  { id: "extraction", label: "Extraction", icon: Map },
  { id: "reconstruction", label: "Reconstruction", icon: Network },
  { id: "resilience", label: "Resilience", icon: ShieldCheck },
  { id: "recovery", label: "Recovery", icon: Wrench },
];

function metricValue(value, suffix = "") {
  if (value === undefined || value === null) return "-";
  const formatted = typeof value === "number" ? value.toLocaleString() : value;
  return `${formatted}${suffix}`;
}

function outputUrl(path) {
  if (!path) return "";
  return path.startsWith("http") ? path : `${API_BASE}${path}`;
}

function StatCard({ label, value, icon: Icon, tone = "default", suffix = "" }) {
  return (
    <div className={`stat-card ${tone}`}>
      <div className="stat-icon">{Icon ? <Icon size={18} /> : <Activity size={18} />}</div>
      <div>
        <span>{label}</span>
        <strong>{metricValue(value, suffix)}</strong>
      </div>
    </div>
  );
}

function OutputPanel({ title, src, children }) {
  return (
    <article className="output-panel">
      <div className="output-media">
        {src ? <img src={src} alt={title} /> : <div className="empty-frame">Output pending</div>}
      </div>
      <div className="output-copy">
        <div className="panel-head">
          <h2>{title}</h2>
          {src ? (
            <a href={src} target="_blank" rel="noreferrer">
              Open
            </a>
          ) : null}
        </div>
        <div className="explain">{children}</div>
      </div>
    </article>
  );
}

function DetailList({ rows }) {
  return (
    <dl>
      {rows.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <Layers size={28} />
      <strong>No output yet</strong>
      <span>Run the related action from the left workflow panel.</span>
    </div>
  );
}

function App() {
  const [page, setPage] = React.useState("overview");
  const [file, setFile] = React.useState(null);
  const [busy, setBusy] = React.useState("");
  const [error, setError] = React.useState("");
  const [log, setLog] = React.useState([]);
  const [state, setState] = React.useState({
    savedFilename: "",
    maskFilename: "",
    reconstructedFilename: "",
    originalPreview: "",
    maskUrl: "",
    graphUrl: "",
    reconstructedUrl: "",
    criticalityUrl: "",
    disasterUrl: "",
    recoveryUrl: "",
    graphMetrics: null,
    comparison: null,
    criticality: null,
    disaster: null,
    recovery: null,
  });

  function pushLog(message) {
    setLog((items) => [message, ...items].slice(0, 6));
  }

  async function request(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, options);
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Request failed");
    }

    return data;
  }

  async function runStep(name, targetPage, callback) {
    setBusy(name);
    setError("");

    try {
      await callback();
      setPage(targetPage);
      pushLog(`${name} completed`);
    } catch (caught) {
      setError(caught.message);
      pushLog(`${name} failed`);
    } finally {
      setBusy("");
    }
  }

  async function uploadImage() {
    if (!file) {
      setError("Choose a satellite image first.");
      return;
    }

    await runStep("Upload", "extraction", async () => {
      const formData = new FormData();
      formData.append("file", file);
      const data = await request("/upload-image", { method: "POST", body: formData });

      setState({
        savedFilename: data.saved_filename,
        maskFilename: "",
        reconstructedFilename: "",
        originalPreview: URL.createObjectURL(file),
        maskUrl: "",
        graphUrl: "",
        reconstructedUrl: "",
        criticalityUrl: "",
        disasterUrl: "",
        recoveryUrl: "",
        graphMetrics: null,
        comparison: null,
        criticality: null,
        disaster: null,
        recovery: null,
      });
    });
  }

  async function extractRoads() {
    if (!state.savedFilename) {
      setError("Upload an image first.");
      return;
    }

    await runStep("Road extraction", "extraction", async () => {
      const data = await request(`/extract-roads/${encodeURIComponent(state.savedFilename)}`, { method: "POST" });
      setState((current) => ({
        ...current,
        maskFilename: data.mask_filename,
        maskUrl: outputUrl(data.mask_url),
      }));
    });
  }

  async function generateGraph() {
    if (!state.maskFilename) {
      setError("Extract roads first.");
      return;
    }

    await runStep("Graph analysis", "extraction", async () => {
      const [metrics, visual] = await Promise.all([
        request(`/generate-graph/${encodeURIComponent(state.maskFilename)}`, { method: "POST" }),
        request(`/visualize-graph/${encodeURIComponent(state.maskFilename)}`, { method: "POST" }),
      ]);

      setState((current) => ({
        ...current,
        graphMetrics: metrics.graph_metrics,
        graphUrl: outputUrl(visual.graph_url),
      }));
    });
  }

  async function reconstructRoads() {
    if (!state.maskFilename) {
      setError("Extract roads first.");
      return;
    }

    await runStep("Reconstruction", "reconstruction", async () => {
      const data = await request(
        `/reconstruct-roads/${encodeURIComponent(state.maskFilename)}?max_gap_pixels=35&min_alignment_score=0.35`,
        { method: "POST" },
      );

      setState((current) => ({
        ...current,
        reconstructedFilename: data.reconstructed_filename,
        reconstructedUrl: outputUrl(data.reconstructed_url),
      }));
    });
  }

  async function compareNetwork() {
    if (!state.maskFilename || !state.reconstructedFilename) {
      setError("Run reconstruction first.");
      return;
    }

    await runStep("Network comparison", "reconstruction", async () => {
      const params = new URLSearchParams({
        original_mask_filename: state.maskFilename,
        reconstructed_mask_filename: state.reconstructedFilename,
      });
      const data = await request(`/compare-network?${params.toString()}`, { method: "POST" });
      setState((current) => ({ ...current, comparison: data.comparison }));
    });
  }

  async function analyzeCriticality() {
    const filename = state.reconstructedFilename || state.maskFilename;

    if (!filename) {
      setError("Create a mask first.");
      return;
    }

    await runStep("Criticality", "resilience", async () => {
      const data = await request(`/criticality/${encodeURIComponent(filename)}?top_k=20&sample_size=300`, {
        method: "POST",
      });
      setState((current) => ({
        ...current,
        criticality: data.metrics,
        criticalityUrl: outputUrl(data.criticality_url),
      }));
    });
  }

  async function simulateDisaster() {
    const filename = state.reconstructedFilename || state.maskFilename;

    if (!filename) {
      setError("Create a mask first.");
      return;
    }

    await runStep("Disaster simulation", "resilience", async () => {
      const data = await request(
        `/simulate-disaster/${encodeURIComponent(filename)}?failure_percent=5&simulation_type=random&seed=42`,
        { method: "POST" },
      );
      setState((current) => ({
        ...current,
        disaster: data.metrics,
        disasterUrl: outputUrl(data.simulation_url),
      }));
    });
  }

  async function recoveryPriority() {
    const filename = state.reconstructedFilename || state.maskFilename;

    if (!filename) {
      setError("Create a mask first.");
      return;
    }

    await runStep("Recovery priority", "recovery", async () => {
      const data = await request(
        `/recovery-priority/${encodeURIComponent(filename)}?failure_percent=5&simulation_type=random&seed=42&top_k=10&candidate_limit=150`,
        { method: "POST" },
      );
      setState((current) => ({
        ...current,
        recovery: data.metrics,
        recoveryUrl: outputUrl(data.recovery_url),
      }));
    });
  }

  const activeMask = state.reconstructedFilename || state.maskFilename;

  const workflow = [
    { label: "Upload", icon: Upload, action: uploadImage },
    { label: "Extract Roads", icon: Map, action: extractRoads },
    { label: "Build Graph", icon: GitBranch, action: generateGraph },
    { label: "Reconstruct", icon: Network, action: reconstructRoads },
    { label: "Compare", icon: BarChart3, action: compareNetwork },
    { label: "Criticality", icon: ShieldCheck, action: analyzeCriticality },
    { label: "Disaster", icon: AlertTriangle, action: simulateDisaster },
    { label: "Recovery", icon: Wrench, action: recoveryPriority },
  ];

  return (
    <main>
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Route size={25} />
          </div>
          <div>
            <h1>StatRoute AI</h1>
            <p>Occlusion-aware road intelligence</p>
          </div>
        </div>

        <nav className="page-nav">
          {pages.map((item) => {
            const Icon = item.icon;
            return (
              <button
                className={page === item.id ? "active" : ""}
                key={item.id}
                onClick={() => setPage(item.id)}
                type="button"
              >
                <Icon size={17} />
                {item.label}
              </button>
            );
          })}
        </nav>

        <label className="upload-zone">
          <ImageUp size={26} />
          <span>{file ? file.name : "Choose satellite image"}</span>
          <input type="file" accept=".jpg,.jpeg,.png,.tif,.tiff" onChange={(event) => setFile(event.target.files[0])} />
        </label>

        <div className="workflow">
          {workflow.map((item) => {
            const Icon = item.icon;
            return (
              <button onClick={item.action} disabled={Boolean(busy)} key={item.label} type="button">
                <Icon size={17} />
                <span>{item.label}</span>
                <ChevronRight size={15} />
              </button>
            );
          })}
        </div>

        <div className="status-box">
          <div>
            <span>Status</span>
            <strong>{busy || "Ready"}</strong>
          </div>
          {busy ? <RefreshCw className="spin" size={20} /> : <Activity size={20} />}
        </div>

        {error ? <div className="error">{error}</div> : null}

        <div className="run-log">
          <h2>Run Log</h2>
          {log.length ? log.map((item) => <p key={item}>{item}</p>) : <p>No actions yet</p>}
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <span>Active mask</span>
            <strong>{activeMask || "None yet"}</strong>
          </div>
          <div>
            <span>Backend</span>
            <strong>{API_BASE}</strong>
          </div>
        </header>

        {page === "overview" ? (
          <OverviewPage state={state} />
        ) : page === "extraction" ? (
          <ExtractionPage state={state} />
        ) : page === "reconstruction" ? (
          <ReconstructionPage state={state} />
        ) : page === "resilience" ? (
          <ResiliencePage state={state} />
        ) : (
          <RecoveryPage state={state} />
        )}
      </section>
    </main>
  );
}

function OverviewPage({ state }) {
  return (
    <>
      <section className="hero-band">
        <div>
          <span>System Snapshot</span>
          <h2>Road extraction, repair, and disruption analysis in one workflow.</h2>
        </div>
      </section>

      <section className="metrics-grid">
        <StatCard label="Graph Nodes" value={state.graphMetrics?.nodes} icon={Network} />
        <StatCard label="Graph Edges" value={state.graphMetrics?.edges} icon={GitBranch} />
        <StatCard label="Components" value={state.graphMetrics?.connected_components} icon={Map} />
        <StatCard label="Resilience Score" value={state.disaster?.resilience_score} icon={ShieldCheck} suffix="%" tone="green" />
      </section>

      <section className="overview-grid">
        <OutputPanel title="Latest Mask" src={state.reconstructedUrl || state.maskUrl}>
          <p>
            This view represents the current road network mask. When reconstruction exists, it becomes the active network
            used for criticality, disaster, and recovery analysis.
          </p>
        </OutputPanel>
        <OutputPanel title="Latest Risk View" src={state.disasterUrl || state.criticalityUrl || state.recoveryUrl}>
          <p>
            This panel shows the newest resilience layer produced by the system: critical locations, simulated road
            failures, or recovery priorities depending on the last analysis you ran.
          </p>
        </OutputPanel>
      </section>
    </>
  );
}

function ExtractionPage({ state }) {
  return (
    <section className="page-stack">
      <div className="page-title">
        <span>Extraction</span>
        <h2>Visible roads are converted into a binary road mask and graph view.</h2>
      </div>

      <OutputPanel title="Original Satellite Image" src={state.originalPreview}>
        <p>The uploaded satellite image is the raw input. Roads may be interrupted by tree cover, shadows, buildings, or noise.</p>
      </OutputPanel>

      <OutputPanel title="Road Mask" src={state.maskUrl}>
        <p>
          The first processing stage highlights detected road-like pixels. White areas are treated as candidate roads for
          graph construction.
        </p>
      </OutputPanel>

      <OutputPanel title="Graph View" src={state.graphUrl}>
        <p>
          The mask is thinned into a skeleton. White lines show road centerlines, red marks endpoints, and yellow marks
          intersection-like points.
        </p>
        <DetailList
          rows={[
            ["Nodes", metricValue(state.graphMetrics?.nodes)],
            ["Edges", metricValue(state.graphMetrics?.edges)],
            ["Connected Components", metricValue(state.graphMetrics?.connected_components)],
            ["Largest Component", metricValue(state.graphMetrics?.largest_component_nodes)],
          ]}
        />
      </OutputPanel>
    </section>
  );
}

function ReconstructionPage({ state }) {
  return (
    <section className="page-stack">
      <div className="page-title">
        <span>Reconstruction</span>
        <h2>Broken road endpoints are linked when the geometry suggests hidden connectivity.</h2>
      </div>

      {state.reconstructedUrl || state.comparison ? (
        <>
          <OutputPanel title="Reconstructed Road Mask" src={state.reconstructedUrl}>
            <p>
              The reconstruction stage searches for nearby road endpoints and joins likely missing links caused by
              occlusions or extraction gaps.
            </p>
          </OutputPanel>

          <section className="analysis-grid">
            <article>
              <h2>Connectivity Change</h2>
              <DetailList
                rows={[
                  ["Status", metricValue(state.comparison?.status)],
                  ["Components Reduced", metricValue(state.comparison?.components_reduced_by)],
                  ["Improvement", metricValue(state.comparison?.connectivity_improvement_percent, "%")],
                  ["Edges Added", metricValue(state.comparison?.edges_added)],
                ]}
              />
            </article>
          </section>
        </>
      ) : (
        <EmptyState />
      )}
    </section>
  );
}

function ResiliencePage({ state }) {
  return (
    <section className="page-stack">
      <div className="page-title">
        <span>Resilience</span>
        <h2>Critical road points and simulated disruptions reveal network weakness.</h2>
      </div>

      <OutputPanel title="Criticality Map" src={state.criticalityUrl}>
        <p>
          Gray lines show the road skeleton. Red and yellow circles identify high-priority road nodes that act like
          important junctions or connection points.
        </p>
        <DetailList
          rows={[
            ["Analyzed Nodes", metricValue(state.criticality?.total_analyzed_nodes)],
            ["Largest Component", metricValue(state.criticality?.largest_component_nodes)],
            ["Top Critical Points", metricValue(state.criticality?.top_k)],
          ]}
        />
      </OutputPanel>

      <OutputPanel title="Disaster Simulation" src={state.disasterUrl}>
        <p>
          A random disruption removes part of the road network. Gray pixels survived the event, while red pixels indicate
          blocked or failed road locations.
        </p>
        <DetailList
          rows={[
            ["Failed Nodes", metricValue(state.disaster?.failed_nodes)],
            ["Connectivity Loss", metricValue(state.disaster?.connectivity_loss_percent, "%")],
            ["Resilience Score", metricValue(state.disaster?.resilience_score, "%")],
            ["After Components", metricValue(state.disaster?.after?.connected_components)],
          ]}
        />
      </OutputPanel>
    </section>
  );
}

function RecoveryPage({ state }) {
  const first = state.recovery?.recovery_priority?.[0];

  return (
    <section className="page-stack">
      <div className="page-title">
        <span>Recovery</span>
        <h2>Repair priorities are ranked by how much connectivity they restore.</h2>
      </div>

      <OutputPanel title="Recovery Priority Map" src={state.recoveryUrl}>
        <p>
          Red marks failed road nodes from the simulated event. Green marks show the highest-priority repair locations
          recommended by StatRoute AI.
        </p>
        <DetailList
          rows={[
            ["Failed Nodes", metricValue(state.recovery?.failed_nodes)],
            ["Evaluated Candidates", metricValue(state.recovery?.evaluated_candidates)],
            ["Top Repair Point", first ? `${first.row}, ${first.col}` : "-"],
            ["Restoration Gain", metricValue(first?.restoration_gain)],
          ]}
        />
      </OutputPanel>
    </section>
  );
}

createRoot(document.getElementById("root")).render(<App />);
