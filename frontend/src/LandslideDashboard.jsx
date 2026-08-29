/**
 * LandslideDashboard.jsx
 *
 * Emergency operations dashboard for landslide risk monitoring across the
 * North Eastern Region (NER) of India.
 *
 * Peer dependencies (host project must install):
 *   react, react-dom          (React 18+ / 19)
 *   react-leaflet@^5  leaflet@^1.9
 *   leaflet.heat@^0.2          -- only needed for the rainfall heatmap layer
 *   leaflet/dist/leaflet.css must be importable (imported below)
 *   Tailwind CSS must be configured in the host project (v3: add this file's
 *   path to `content`; v4: nothing extra needed beyond `@import "tailwindcss"`
 *   in your global stylesheet, since Tailwind v4 auto-detects source files).
 *
 * This component is self-contained and ships with realistic sample data
 * (`DEFAULT_*` exports below) so it renders immediately with no wiring.
 * Everything is overridable via props -- swap in your real district
 * GeoJSON, incident feed, road network, rainfall telemetry, and priority
 * queue from your own API/state layer. The sample GeoJSON polygons are
 * simplified illustrative shapes centered on real district locations, NOT
 * authoritative administrative boundaries -- replace with real boundary
 * data (e.g. from a GIS/Survey of India source) before using this
 * operationally.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import {
  MapContainer,
  TileLayer,
  GeoJSON,
  Marker,
  Popup,
  Polyline,
  useMap,
} from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import 'leaflet.heat'

// --------------------------------------------------------------------------
// Map basics
// --------------------------------------------------------------------------

const NER_CENTER = [26.3, 92.4] // roughly between Assam, Meghalaya & Sikkim
const NER_DEFAULT_ZOOM = 7

// CARTO dark basemap -- free for reasonable-traffic use with attribution.
// Swap for Mapbox/MapTiler/your own tile server in production.
const DARK_TILE_URL =
  'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
const DARK_TILE_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'

const RISK_LEVELS = {
  High: { stroke: '#f87171', fill: '#ef4444', badge: 'bg-red-500/20 text-red-300 border-red-500/40' },
  Medium: { stroke: '#fbbf24', fill: '#eab308', badge: 'bg-amber-500/20 text-amber-300 border-amber-500/40' },
  Low: { stroke: '#4ade80', fill: '#22c55e', badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40' },
}

function riskBandForScore(score) {
  if (score >= 70) return 'High'
  if (score >= 40) return 'Medium'
  return 'Low'
}

// --------------------------------------------------------------------------
// Sample data -- replace via props with real feeds. Shapes/coordinates are
// simplified and illustrative, anchored near real NER district locations.
// --------------------------------------------------------------------------

function squarePolygon([lat, lng], halfSize = 0.14) {
  return [
    [
      [lng - halfSize, lat - halfSize],
      [lng + halfSize, lat - halfSize],
      [lng + halfSize, lat + halfSize],
      [lng - halfSize, lat + halfSize],
      [lng - halfSize, lat - halfSize],
    ],
  ]
}

const DISTRICT_SEEDS = [
  { district: 'East Sikkim', state: 'Sikkim', center: [27.33, 88.62], risk_level: 'High', risk_score: 88 },
  { district: 'North Sikkim', state: 'Sikkim', center: [27.51, 88.53], risk_level: 'High', risk_score: 82 },
  { district: 'East Khasi Hills', state: 'Meghalaya', center: [25.57, 91.88], risk_level: 'High', risk_score: 79 },
  { district: 'Dima Hasao', state: 'Assam', center: [25.17, 93.02], risk_level: 'Medium', risk_score: 58 },
  { district: 'West Kameng', state: 'Arunachal Pradesh', center: [27.26, 92.4], risk_level: 'Medium', risk_score: 51 },
  { district: 'Aizawl', state: 'Mizoram', center: [23.73, 92.72], risk_level: 'Medium', risk_score: 46 },
  { district: 'Kohima', state: 'Nagaland', center: [25.67, 94.11], risk_level: 'Low', risk_score: 28 },
  { district: 'Senapati', state: 'Manipur', center: [25.27, 94.02], risk_level: 'Low', risk_score: 22 },
]

export const DEFAULT_RISK_POLYGONS = {
  type: 'FeatureCollection',
  features: DISTRICT_SEEDS.map((d) => ({
    type: 'Feature',
    properties: { district: d.district, state: d.state, risk_level: d.risk_level, risk_score: d.risk_score },
    geometry: { type: 'Polygon', coordinates: squarePolygon(d.center) },
  })),
}

const REPORT_TYPE_META = {
  active_mudslide: { label: 'Active Mudslide', color: '#f97316', symbol: '⚠' },
  blocked_road: { label: 'Blocked Road', color: '#ef4444', symbol: '✕' },
  road_crack: { label: 'Road Crack', color: '#eab308', symbol: '≈' },
  other: { label: 'Other Report', color: '#38bdf8', symbol: '•' },
}

export const DEFAULT_INCIDENT_REPORTS = [
  { id: 'RPT-1042', type: 'active_mudslide', lat: 27.34, lng: 88.6, reporter: 'Field volunteer, Gangtok', description: 'Debris flow across NH10 near the 4th Mile bend, road impassable.', timestamp: '2026-08-28T04:12:00+05:30', verified: true },
  { id: 'RPT-1041', type: 'blocked_road', lat: 25.58, lng: 91.9, reporter: 'District control room, Shillong', description: 'Boulders blocking Shillong-Guwahati road near Nongpoh.', timestamp: '2026-08-28T02:47:00+05:30', verified: true },
  { id: 'RPT-1040', type: 'road_crack', lat: 25.18, lng: 93.0, reporter: 'PWD inspector', description: 'Longitudinal crack widening on hill-cut section, Haflong bypass.', timestamp: '2026-08-27T21:30:00+05:30', verified: false },
  { id: 'RPT-1039', type: 'active_mudslide', lat: 27.5, lng: 88.55, reporter: 'Citizen report (app)', description: 'Slope failure above village access road, North Sikkim.', timestamp: '2026-08-27T19:05:00+05:30', verified: false },
  { id: 'RPT-1038', type: 'blocked_road', lat: 23.75, lng: 92.7, reporter: 'District control room, Aizawl', description: 'Fallen tree and mud across Aizawl-Lunglei highway.', timestamp: '2026-08-27T14:22:00+05:30', verified: true },
]

export const DEFAULT_ROAD_NETWORK = [
  { id: 'ROAD-NH10', name: 'NH10 (Sevoke - Gangtok)', status: 'blocked', coordinates: [[26.9, 88.45], [27.05, 88.5], [27.2, 88.55], [27.33, 88.62]] },
  { id: 'ROAD-SHL-GHY', name: 'Shillong - Guwahati Road', status: 'blocked', coordinates: [[26.14, 91.73], [25.9, 91.8], [25.57, 91.88]] },
  { id: 'ROAD-NH27', name: 'NH27 (Silchar - Haflong)', status: 'at_risk', coordinates: [[24.83, 92.78], [25.0, 92.9], [25.17, 93.02]] },
  { id: 'ROAD-IMP-KHM', name: 'Imphal - Kohima Road', status: 'open', coordinates: [[24.82, 93.94], [25.2, 94.0], [25.67, 94.11]] },
  { id: 'ROAD-AIZ-LGL', name: 'Aizawl - Lunglei Highway', status: 'at_risk', coordinates: [[23.73, 92.72], [23.4, 92.75], [22.88, 92.74]] },
]

const ROAD_STATUS_STYLE = {
  blocked: { color: '#ef4444', dashArray: '6 5', weight: 5 },
  at_risk: { color: '#f59e0b', dashArray: '2 6', weight: 4 },
  open: { color: '#38bdf8', dashArray: null, weight: 3 },
}

export const DEFAULT_RAINFALL_POINTS = [
  { lat: 27.33, lng: 88.62, intensity: 0.95 },
  { lat: 27.4, lng: 88.58, intensity: 0.8 },
  { lat: 25.57, lng: 91.88, intensity: 0.85 },
  { lat: 25.65, lng: 91.95, intensity: 0.6 },
  { lat: 25.17, lng: 93.02, intensity: 0.55 },
  { lat: 26.14, lng: 91.73, intensity: 0.4 },
  { lat: 23.73, lng: 92.72, intensity: 0.5 },
  { lat: 25.67, lng: 94.11, intensity: 0.25 },
  { lat: 27.1, lng: 93.62, intensity: 0.45 },
]

export const DEFAULT_PRIORITY_QUEUE = [
  { id: 'Q-1', name: 'Rangpo - NH10 corridor', type: 'Road', district: 'East Sikkim', riskScore: 91, peopleAtRisk: 1200, status: 'Dispatched', lat: 27.34, lng: 88.6 },
  { id: 'Q-2', name: 'Mawkynrew village cluster', type: 'Village', district: 'East Khasi Hills', riskScore: 84, peopleAtRisk: 3400, status: 'Pending', lat: 25.52, lng: 91.83 },
  { id: 'Q-3', name: 'Chungthang approach road', type: 'Road', district: 'North Sikkim', riskScore: 80, peopleAtRisk: 900, status: 'Pending', lat: 27.5, lng: 88.55 },
  { id: 'Q-4', name: 'Haflong bypass hill-cut', type: 'Road', district: 'Dima Hasao', riskScore: 66, peopleAtRisk: 2100, status: 'Pending', lat: 25.17, lng: 93.02 },
  { id: 'Q-5', name: 'Nongpoh market settlement', type: 'Village', district: 'East Khasi Hills', riskScore: 61, peopleAtRisk: 5200, status: 'Monitoring', lat: 25.9, lng: 91.87 },
  { id: 'Q-6', name: 'Lunglei fringe settlements', type: 'Village', district: 'Aizawl', riskScore: 47, peopleAtRisk: 1800, status: 'Monitoring', lat: 23.5, lng: 92.75 },
  { id: 'Q-7', name: 'Dirang - Bomdila stretch', type: 'Road', district: 'West Kameng', riskScore: 39, peopleAtRisk: 700, status: 'Resolved', lat: 27.35, lng: 92.35 },
]

// --------------------------------------------------------------------------
// Small presentational helpers
// --------------------------------------------------------------------------

function formatTimeAgo(isoString) {
  const diffMs = Date.now() - new Date(isoString).getTime()
  const hrs = diffMs / 3_600_000
  if (hrs < 1) return `${Math.max(1, Math.round(hrs * 60))}m ago`
  if (hrs < 24) return `${Math.round(hrs)}h ago`
  return `${Math.round(hrs / 24)}d ago`
}

function reportDivIcon(type) {
  const meta = REPORT_TYPE_META[type] || REPORT_TYPE_META.other
  return L.divIcon({
    className: 'landslide-dashboard-report-marker',
    html: `<div style="
        width:26px;height:26px;border-radius:9999px;
        background:${meta.color};border:2px solid rgba(15,23,42,0.9);
        box-shadow:0 0 0 2px rgba(255,255,255,0.65), 0 2px 6px rgba(0,0,0,0.5);
        display:flex;align-items:center;justify-content:center;
        color:#0f172a;font-size:13px;font-weight:700;line-height:1;
      ">${meta.symbol}</div>`,
    iconSize: [26, 26],
    iconAnchor: [13, 13],
    popupAnchor: [0, -14],
  })
}

function riskPolygonStyle(feature) {
  const level = RISK_LEVELS[feature?.properties?.risk_level] ? feature.properties.risk_level : 'Low'
  const palette = RISK_LEVELS[level]
  return {
    color: palette.stroke,
    weight: 2,
    fillColor: palette.fill,
    fillOpacity: 0.32,
  }
}

// --------------------------------------------------------------------------
// Map sub-layers that need access to the underlying Leaflet map instance
// --------------------------------------------------------------------------

function RainfallHeatLayer({ points, visible }) {
  const map = useMap()

  useEffect(() => {
    if (!visible || !points?.length) return undefined

    const heatLayer = L.heatLayer(
      points.map((p) => [p.lat, p.lng, p.intensity]),
      { radius: 32, blur: 24, maxZoom: 10, minOpacity: 0.25, gradient: { 0.2: '#1d4ed8', 0.5: '#22c55e', 0.75: '#eab308', 1: '#ef4444' } },
    )
    heatLayer.addTo(map)
    return () => {
      map.removeLayer(heatLayer)
    }
  }, [map, points, visible])

  return null
}

/** Keeps the Leaflet map's internal size in sync with its container -- guards
 * against the common "map renders half-cut-off" bug when this component is
 * mounted inside a flex/grid layout that resizes after first paint. */
function MapSizeSync() {
  const map = useMap()
  useEffect(() => {
    const invalidate = () => map.invalidateSize()
    const id = window.setTimeout(invalidate, 150)
    window.addEventListener('resize', invalidate)
    return () => {
      window.clearTimeout(id)
      window.removeEventListener('resize', invalidate)
    }
  }, [map])
  return null
}

// --------------------------------------------------------------------------
// Sidebar pieces
// --------------------------------------------------------------------------

function StatCard({ label, value, accentClass }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-400">{label}</p>
      <p className={`mt-1 text-2xl font-semibold ${accentClass}`}>{value}</p>
    </div>
  )
}

function FilterToggle({ label, description, checked, onChange }) {
  return (
    <label className="flex cursor-pointer items-start justify-between gap-3 rounded-lg px-2 py-2 hover:bg-slate-800/60">
      <span>
        <span className="block text-sm font-medium text-slate-200">{label}</span>
        {description && <span className="block text-xs text-slate-500">{description}</span>}
      </span>
      <span
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative mt-0.5 h-5 w-9 shrink-0 rounded-full transition-colors ${checked ? 'bg-emerald-500' : 'bg-slate-700'}`}
      >
        <span
          className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform ${checked ? 'translate-x-4' : 'translate-x-0.5'}`}
        />
      </span>
      {/* Hidden native checkbox keeps this keyboard/screen-reader accessible */}
      <input
        type="checkbox"
        className="sr-only"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
    </label>
  )
}

function RiskLegend() {
  return (
    <div className="flex flex-wrap gap-3 rounded-lg border border-slate-800 bg-slate-900/60 p-3 text-xs">
      {Object.entries(RISK_LEVELS).map(([level, palette]) => (
        <span key={level} className="flex items-center gap-1.5 text-slate-300">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: palette.fill }} />
          {level}
        </span>
      ))}
    </div>
  )
}

// --------------------------------------------------------------------------
// Emergency priority queue table
// --------------------------------------------------------------------------

const STATUS_STYLE = {
  Pending: 'bg-slate-700/60 text-slate-200',
  Dispatched: 'bg-sky-500/20 text-sky-300 border border-sky-500/40',
  Monitoring: 'bg-amber-500/20 text-amber-300 border border-amber-500/40',
  Resolved: 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40',
}

function PriorityQueueTable({ items, onLocate }) {
  const sorted = useMemo(() => [...items].sort((a, b) => b.riskScore - a.riskScore), [items])

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-slate-800 px-4 py-2">
        <h2 className="text-sm font-semibold text-slate-200">Emergency Priority Queue</h2>
        <span className="text-xs text-slate-500">{sorted.length} locations &middot; sorted by AI risk score</span>
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        <table className="w-full min-w-[640px] text-left text-sm">
          <thead className="sticky top-0 bg-slate-900 text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-4 py-2 font-medium">#</th>
              <th className="px-4 py-2 font-medium">Location</th>
              <th className="px-4 py-2 font-medium">District</th>
              <th className="px-4 py-2 font-medium">Risk Score</th>
              <th className="px-4 py-2 font-medium">People at Risk</th>
              <th className="px-4 py-2 font-medium">Status</th>
              <th className="px-4 py-2 font-medium" />
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {sorted.map((item, i) => {
              const band = riskBandForScore(item.riskScore)
              return (
                <tr key={item.id} className="text-slate-300 hover:bg-slate-800/50">
                  <td className="px-4 py-2 text-slate-500">{i + 1}</td>
                  <td className="px-4 py-2">
                    <span className="font-medium text-slate-100">{item.name}</span>
                    <span className="ml-2 text-xs text-slate-500">{item.type}</span>
                  </td>
                  <td className="px-4 py-2 text-slate-400">{item.district}</td>
                  <td className="px-4 py-2">
                    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-semibold ${RISK_LEVELS[band].badge}`}>
                      {item.riskScore}
                    </span>
                  </td>
                  <td className="px-4 py-2 text-slate-400">{item.peopleAtRisk.toLocaleString('en-IN')}</td>
                  <td className="px-4 py-2">
                    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${STATUS_STYLE[item.status] || STATUS_STYLE.Pending}`}>
                      {item.status}
                    </span>
                  </td>
                  <td className="px-4 py-2 text-right">
                    <button
                      type="button"
                      onClick={() => onLocate?.(item)}
                      className="rounded-md border border-slate-700 px-2 py-1 text-xs font-medium text-slate-300 hover:border-sky-500 hover:text-sky-300"
                    >
                      Locate
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// --------------------------------------------------------------------------
// Main component
// --------------------------------------------------------------------------

export default function LandslideDashboard({
  riskPolygons = DEFAULT_RISK_POLYGONS,
  incidentReports = DEFAULT_INCIDENT_REPORTS,
  roadNetwork = DEFAULT_ROAD_NETWORK,
  rainfallPoints = DEFAULT_RAINFALL_POINTS,
  priorityQueue = DEFAULT_PRIORITY_QUEUE,
  center = NER_CENTER,
  zoom = NER_DEFAULT_ZOOM,
}) {
  const [showRainfall, setShowRainfall] = useState(false)
  const [showRoads, setShowRoads] = useState(true)
  const [showReports, setShowReports] = useState(true)
  const mapRef = useRef(null)

  const stats = useMemo(() => {
    const highRiskDistricts = riskPolygons.features.filter((f) => f.properties.risk_level === 'High').length
    const blockedHighways = roadNetwork.filter((r) => r.status === 'blocked').length
    const activeFieldReports = incidentReports.length
    return { highRiskDistricts, blockedHighways, activeFieldReports }
  }, [riskPolygons, roadNetwork, incidentReports])

  function handleLocate(item) {
    mapRef.current?.flyTo([item.lat, item.lng], 11, { duration: 0.8 })
  }

  function onEachDistrictFeature(feature, layer) {
    const { district, state, risk_level: riskLevel, risk_score: riskScore } = feature.properties
    layer.bindPopup(
      `<div style="font-family:inherit">
         <strong>${district}</strong><br/>
         <span style="opacity:.75">${state}</span><br/>
         Risk level: <strong>${riskLevel}</strong> (score ${riskScore})
       </div>`,
    )
    layer.on({
      mouseover: (e) => e.target.setStyle({ fillOpacity: 0.55 }),
      mouseout: (e) => e.target.setStyle({ fillOpacity: 0.32 }),
    })
  }

  return (
    <div className="flex h-screen w-full flex-col bg-slate-950 text-slate-100">
      {/* Header */}
      <header className="flex shrink-0 items-center justify-between border-b border-slate-800 bg-slate-900/80 px-4 py-3">
        <div>
          <h1 className="text-lg font-semibold text-slate-50">NER Landslide Early Warning Dashboard</h1>
          <p className="text-xs text-slate-500">North Eastern Region &middot; live situational awareness</p>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
          Live
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        {/* Sidebar */}
        <aside className="flex w-80 shrink-0 flex-col gap-4 overflow-y-auto border-r border-slate-800 bg-slate-900/40 p-4">
          <section>
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Live Stats</h2>
            <div className="grid grid-cols-1 gap-2">
              <StatCard label="High-Risk Districts" value={stats.highRiskDistricts} accentClass="text-red-400" />
              <StatCard label="Blocked Highways" value={stats.blockedHighways} accentClass="text-amber-400" />
              <StatCard label="Active Field Reports" value={stats.activeFieldReports} accentClass="text-sky-400" />
            </div>
          </section>

          <section>
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Map Layers</h2>
            <div className="flex flex-col rounded-lg border border-slate-800">
              <FilterToggle
                label="Rainfall Heatmap Layer"
                description="24h rainfall intensity"
                checked={showRainfall}
                onChange={setShowRainfall}
              />
              <FilterToggle
                label="Road Connectivity Network"
                description="Highways & access roads"
                checked={showRoads}
                onChange={setShowRoads}
              />
              <FilterToggle
                label="Citizen Reports"
                description="Crowdsourced incident pins"
                checked={showReports}
                onChange={setShowReports}
              />
            </div>
          </section>

          <section>
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Risk Legend</h2>
            <RiskLegend />
          </section>
        </aside>

        {/* Map + priority queue */}
        <main className="flex min-h-0 flex-1 flex-col">
          <div className="relative min-h-0 flex-1">
            <MapContainer
              ref={mapRef}
              center={center}
              zoom={zoom}
              className="h-full w-full"
              style={{ background: '#0f172a' }}
            >
              <TileLayer url={DARK_TILE_URL} attribution={DARK_TILE_ATTRIBUTION} />

              <GeoJSON
                key={riskPolygons.features.length}
                data={riskPolygons}
                style={riskPolygonStyle}
                onEachFeature={onEachDistrictFeature}
              />

              {showRoads &&
                roadNetwork.map((road) => (
                  <Polyline
                    key={road.id}
                    positions={road.coordinates}
                    pathOptions={ROAD_STATUS_STYLE[road.status] || ROAD_STATUS_STYLE.open}
                  >
                    <Popup>
                      <strong>{road.name}</strong>
                      <br />
                      Status: {road.status.replace('_', ' ')}
                    </Popup>
                  </Polyline>
                ))}

              {showReports &&
                incidentReports.map((report) => (
                  <Marker key={report.id} position={[report.lat, report.lng]} icon={reportDivIcon(report.type)}>
                    <Popup>
                      <div className="space-y-1">
                        <strong>{(REPORT_TYPE_META[report.type] || REPORT_TYPE_META.other).label}</strong>
                        <p style={{ margin: 0 }}>{report.description}</p>
                        <p style={{ margin: 0, opacity: 0.7, fontSize: '0.8em' }}>
                          {report.reporter} &middot; {formatTimeAgo(report.timestamp)}
                          {report.verified ? ' ✓ verified' : ' (unverified)'}
                        </p>
                      </div>
                    </Popup>
                  </Marker>
                ))}

              <RainfallHeatLayer points={rainfallPoints} visible={showRainfall} />
              <MapSizeSync />
            </MapContainer>
          </div>

          {/* Emergency priority queue */}
          <div className="h-64 shrink-0 border-t border-slate-800 bg-slate-950">
            <PriorityQueueTable items={priorityQueue} onLocate={handleLocate} />
          </div>
        </main>
      </div>
    </div>
  )
}
