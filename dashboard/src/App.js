import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const SERVER_URL = "http://127.0.0.1:9000";

function App() {
  const [status, setStatus] = useState({
    current_round: 0,
    active_clients: 0,
    pending_updates: 0,
    clients: {},
    history: [],
  });

  const [metrics, setMetrics] = useState({
    rounds: [],
    accuracy: [],
    loss: [],
  });

  // Poll server every 2 seconds
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch(`${SERVER_URL}/status`);
        const data = await response.json();
        setStatus(data);
      } catch (error) {
        console.error("Failed to fetch status:", error);
      }
    };

    const fetchMetrics = async () => {
      try {
        const response = await fetch(`${SERVER_URL}/metrics`);
        const data = await response.json();
        setMetrics(data);
      } catch (error) {
        console.error("Failed to fetch metrics:", error);
      }
    };

    fetchStatus();
    fetchMetrics();

    const statusInterval = setInterval(fetchStatus, 2000);
    const metricsInterval = setInterval(fetchMetrics, 5000);

    return () => {
      clearInterval(statusInterval);
      clearInterval(metricsInterval);
    };
  }, []);

  // Prepare chart data
  const chartData = metrics.rounds.map((round, idx) => ({
    round,
    accuracy: metrics.accuracy[idx] * 100, // Convert to percentage
    loss: metrics.loss[idx],
  }));

  return (
    <div
      style={{
        padding: "20px",
        fontFamily: "Arial, sans-serif",
        backgroundColor: "#f5f5f5",
        minHeight: "100vh",
      }}
    >
      <h1
        style={{
          color: "#333",
          borderBottom: "3px solid #4CAF50",
          paddingBottom: "10px",
        }}
      >
        🤖 Federated Learning Dashboard
      </h1>

      {/* Top Stats Row */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: "20px",
          marginBottom: "30px",
        }}
      >
        <StatCard
          title="Current Round"
          value={status.current_round}
          color="#4CAF50"
          icon="🔄"
        />
        <StatCard
          title="Active Clients"
          value={status.active_clients}
          color="#2196F3"
          icon="📱"
        />
        <StatCard
          title="Pending Updates"
          value={status.pending_updates}
          color="#FF9800"
          icon="⏳"
        />
      </div>

      {/* Charts Row */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr",
          gap: "20px",
          marginBottom: "30px",
        }}
      >
        <ChartCard title="Model Accuracy Over Rounds">
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="round"
                  label={{
                    value: "Round",
                    position: "insideBottom",
                    offset: -5,
                  }}
                />
                <YAxis
                  label={{
                    value: "Accuracy (%)",
                    angle: -90,
                    position: "insideLeft",
                  }}
                />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#4CAF50"
                  strokeWidth={2}
                  dot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ textAlign: "center", color: "#999", padding: "60px" }}>
              Waiting for training data...
            </p>
          )}
        </ChartCard>
      </div>

      {/* Active Clients Table */}
      <div
        style={{
          backgroundColor: "white",
          padding: "20px",
          borderRadius: "8px",
          boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
        }}
      >
        <h2 style={{ marginTop: 0, color: "#333" }}>📊 Active Clients</h2>
        {Object.keys(status.clients).length === 0 ? (
          <p style={{ color: "#999", textAlign: "center", padding: "20px" }}>
            No clients connected yet. Start some clients to see them here!
          </p>
        ) : (
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ backgroundColor: "#f0f0f0", textAlign: "left" }}>
                <th style={{ padding: "12px", borderBottom: "2px solid #ddd" }}>
                  Client ID
                </th>
                <th style={{ padding: "12px", borderBottom: "2px solid #ddd" }}>
                  Status
                </th>
                <th style={{ padding: "12px", borderBottom: "2px solid #ddd" }}>
                  Data Size
                </th>
                <th style={{ padding: "12px", borderBottom: "2px solid #ddd" }}>
                  Last Update
                </th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(status.clients).map(([clientId, clientInfo]) => (
                <tr key={clientId} style={{ borderBottom: "1px solid #eee" }}>
                  <td style={{ padding: "12px" }}>
                    <span
                      style={{
                        backgroundColor: "#4CAF50",
                        color: "white",
                        padding: "4px 8px",
                        borderRadius: "4px",
                        fontSize: "12px",
                        fontWeight: "bold",
                      }}
                    >
                      Client {clientId}
                    </span>
                  </td>
                  <td style={{ padding: "12px" }}>
                    <StatusBadge status={clientInfo.status} />
                  </td>
                  <td style={{ padding: "12px" }}>
                    {clientInfo.data_size} samples
                  </td>
                  <td
                    style={{ padding: "12px", fontSize: "14px", color: "#666" }}
                  >
                    {new Date(clientInfo.timestamp).toLocaleTimeString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Instructions */}
      <div
        style={{
          marginTop: "30px",
          padding: "20px",
          backgroundColor: "#e3f2fd",
          borderRadius: "8px",
          borderLeft: "4px solid #2196F3",
        }}
      >
        <h3 style={{ marginTop: 0 }}>💡 How to Use</h3>
        <ol style={{ lineHeight: "1.8" }}>
          <li>
            Make sure the server is running:{" "}
            <code>python server/aggregator.py</code>
          </li>
          <li>
            Start clients:{" "}
            <code>python clients/client.py 1 --num-clients 10 --non-iid</code>
          </li>
          <li>
            Open more terminals and start more clients with different IDs (2, 3,
            4, etc.)
          </li>
          <li>
            Watch the dashboard update in real-time as clients train and submit
            updates!
          </li>
        </ol>
      </div>
    </div>
  );
}

// Reusable Components
function StatCard({ title, value, color, icon }) {
  return (
    <div
      style={{
        backgroundColor: "white",
        padding: "20px",
        borderRadius: "8px",
        boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
        borderLeft: `4px solid ${color}`,
      }}
    >
      <div style={{ fontSize: "32px", marginBottom: "10px" }}>{icon}</div>
      <div style={{ fontSize: "14px", color: "#666", marginBottom: "5px" }}>
        {title}
      </div>
      <div style={{ fontSize: "36px", fontWeight: "bold", color }}>{value}</div>
    </div>
  );
}

function ChartCard({ title, children }) {
  return (
    <div
      style={{
        backgroundColor: "white",
        padding: "20px",
        borderRadius: "8px",
        boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
      }}
    >
      <h2 style={{ marginTop: 0, color: "#333" }}>{title}</h2>
      {children}
    </div>
  );
}

function StatusBadge({ status }) {
  const statusColors = {
    submitted: "#4CAF50",
    training: "#FF9800",
    idle: "#9E9E9E",
    error: "#F44336",
  };

  return (
    <span
      style={{
        backgroundColor: statusColors[status] || "#9E9E9E",
        color: "white",
        padding: "4px 12px",
        borderRadius: "12px",
        fontSize: "12px",
        fontWeight: "bold",
        textTransform: "uppercase",
      }}
    >
      {status}
    </span>
  );
}

export default App;
